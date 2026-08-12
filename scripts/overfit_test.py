"""Overfit test: la rete deve memorizzare 512 posizioni (Gate 4, criterio #1).

    python scripts/overfit_test.py
    python scripts/overfit_test.py --positions 512 --steps 400

## A cosa serve

> La rete deve raggiungere ~100% di accuracy su 512 posizioni fisse. Se non ci riesce
> c'e un bug, non un problema di capacita. Da fare *prima* del training vero.

Con 12 milioni di parametri e 512 esempi, memorizzarli e banale — una rete sana ci
arriva in qualche centinaio di passi. Se **non** ci riesce, il problema non e la
difficolta degli scacchi: e che qualcosa nella catena non funziona.

Costa due minuti e copre l'intera pipeline:

| Se fallisce | Il sospetto e |
|---|---|
| la loss non scende affatto | gradienti non collegati, learning rate assurdo, optimizer che non vede i parametri |
| la loss scende ma la top-1 no | etichette scollegate dagli input, mascheratura sbagliata |
| loss a NaN | esplosione numerica, AMP senza scaler, divisione per zero nella loss |
| va bene ma lentissimo | learning rate troppo basso, o warmup piu lungo del test |

E la differenza fra scoprire un bug in due minuti e scoprirlo dopo una notte di training.

## Perche senza augmentation e senza shuffle

Qui si vuole **esattamente** la memorizzazione: stesse posizioni, stesso ordine, nessuna
variazione. Se ci fosse qualcosa di casuale, un fallimento sarebbe ambiguo.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch  # noqa: E402

from chessbot.model.network import ChessNet, NetworkConfig  # noqa: E402
from chessbot.training.dataset import ChessPositions, FixedPositions, make_loader  # noqa: E402
from chessbot.training.loop import TrainConfig, build_optimizer, move_batch  # noqa: E402
from chessbot.training.loss import LossWeights, accuracy_metrics, compute_loss  # noqa: E402
from chessbot.utils.seed import set_seed  # noqa: E402

_COLOR = sys.stdout.isatty()
GREEN, RED, GREY, BOLD, RESET = (
    ("\033[32m", "\033[31m", "\033[90m", "\033[1m", "\033[0m") if _COLOR else ("",) * 5
)

# Soglia del Gate 4. Non 100% esatto: con label smoothing 0.05 e 512 posizioni, qualche
# campione ambiguo puo restare. Sotto il 95% pero c'e un bug.
TARGET_TOP1 = 0.95


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "processed")
    ap.add_argument("--split", default="val", help="da quale split pescare le posizioni")
    ap.add_argument("--positions", type=int, default=512)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"{BOLD}Overfit test — Gate 4, criterio #1{RESET}")
    print(f"  {args.positions} posizioni fisse dallo split {args.split}")
    print(f"  {args.steps} step, batch {args.batch_size}, lr {args.lr}")
    print(f"  device: {device}")
    print()

    source = ChessPositions(args.data, args.split)
    fixed = FixedPositions(source, args.positions, seed=args.seed)
    loader = make_loader(
        fixed,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # i dati sono gia in RAM: i worker aggiungerebbero solo overhead
        seed=args.seed,
    )

    model = ChessNet(NetworkConfig(channels=args.channels, blocks=args.blocks)).to(device)
    print(f"  {model.describe()}")
    print()

    # Niente warmup: con 400 step un warmup da 2000 lascerebbe il LR quasi a zero per
    # tutto il test, e il fallimento sarebbe un artefatto dello scheduler.
    config = TrainConfig(lr=args.lr, warmup_steps=1, weight_decay=0.0)
    optimizer = build_optimizer(model, config)
    weights = LossWeights()

    model.train()
    started = time.perf_counter()
    step = 0
    history: list[tuple[int, float, float]] = []

    print(f"  {'step':>6} {'loss':>9} {'policy':>9} {'top-1':>8} {'top-5':>8}")
    print("  " + "-" * 46)

    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            batch = move_batch(batch, device)

            policy_logits, wdl_logits = model(batch["planes"])
            parts = compute_loss(policy_logits, wdl_logits, batch, weights)

            optimizer.zero_grad(set_to_none=True)
            parts.total.backward()
            optimizer.step()

            step += 1

            if step % 50 == 0 or step == 1:
                metrics = accuracy_metrics(policy_logits, wdl_logits, batch)
                history.append((step, parts.total.item(), metrics["top1"]))
                print(
                    f"  {step:>6} {parts.total.item():>9.4f} {parts.policy.item():>9.4f} "
                    f"{100 * metrics['top1']:>7.1f}% {100 * metrics['top5']:>7.1f}%",
                    flush=True,
                )

    elapsed = time.perf_counter() - started

    # Valutazione finale sull'INTERO insieme fisso, non sull'ultimo batch: e la misura
    # che conta, e un singolo batch fortunato non deve poter far passare il test.
    model.eval()
    totals = {"top1": 0.0, "top5": 0.0, "wdl_acc": 0.0}
    n = 0
    with torch.no_grad():
        for batch in make_loader(fixed, batch_size=args.batch_size, shuffle=False, num_workers=0):
            batch = move_batch(batch, device)
            policy_logits, wdl_logits = model(batch["planes"])
            for k, v in accuracy_metrics(policy_logits, wdl_logits, batch).items():
                totals[k] += v
            n += 1
    final = {k: v / max(n, 1) for k, v in totals.items()}

    print()
    print(f"{BOLD}Risultato su tutte le {len(fixed)} posizioni{RESET}")
    print(f"  top-1   {100 * final['top1']:.2f}%")
    print(f"  top-5   {100 * final['top5']:.2f}%")
    print(f"  wdl     {100 * final['wdl_acc']:.2f}%")
    print(f"  {GREY}{elapsed:.0f}s, {args.steps / elapsed:.1f} step/s{RESET}")

    print()
    ok = final["top1"] >= TARGET_TOP1
    if ok:
        print(f"  {GREEN}PASS — la rete memorizza le posizioni.{RESET}")
        print("  La catena dati -> rete -> loss -> optimizer funziona. Si puo allenare.")
    else:
        print(
            f"  {RED}FAIL — top-1 {100 * final['top1']:.1f}%, attesa >= {100 * TARGET_TOP1:.0f}%.{RESET}"
        )
        print("  C'e un BUG, non un limite di capacita: 12M parametri memorizzano")
        print("  512 esempi senza sforzo. Controllare, in quest'ordine:")
        first_loss = history[0][1] if history else float("nan")
        last_loss = history[-1][1] if history else float("nan")
        if not (last_loss < first_loss):
            print(f"    - la loss NON scende ({first_loss:.3f} -> {last_loss:.3f}):")
            print("      gradienti scollegati, o l'optimizer non vede i parametri")
        else:
            print(f"    - la loss scende ({first_loss:.3f} -> {last_loss:.3f}) ma la top-1 no:")
            print("      etichette scollegate dagli input, o mascheratura sbagliata")
        print("    - provare --steps 1000 se la curva sembrava ancora in discesa")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
