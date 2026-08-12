"""Training supervisionato — lo script che si lancia per allenare (Stadio 4).

    python scripts/train.py                        parte o RIPRENDE automaticamente
    python scripts/train.py --epochs 12
    python scripts/train.py --run prova --max-steps 200    smoke test
    python scripts/train.py --fresh                 ricomincia da zero, ignorando i checkpoint

## La ripresa e automatica

Se `runs/<nome>/last.pt` esiste, lo script riprende da li: pesi, optimizer, scheduler,
scaler, epoca, step e stato dei generatori casuali. **Non serve ricordarsi un flag.**

Vuol dire che dopo un crash, uno standby o un riavvio basta rilanciare lo stesso comando.
Per ricominciare davvero da zero serve `--fresh`, che e esplicito di proposito: il caso
pericoloso e sovrascrivere per sbaglio una notte di training, non doverlo ripetere.

## Cosa resta su disco

    runs/<nome>/
      last.pt          per riprendere — salvato ogni 2000 step
      best.pt          per giocare — la top-1 di validation piu alta vista
      history.jsonl    ogni valutazione, append-only

`last.pt` e `best.pt` sono file diversi perche l'ultimo checkpoint non e
necessariamente il migliore: se la rete peggiora nelle ultime epoche, `best.pt` conserva
comunque il punto buono.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch  # noqa: E402

from chessbot.model.network import ChessNet, NetworkConfig  # noqa: E402
from chessbot.training.checkpoint import (  # noqa: E402
    TrainingState,
    inspect_checkpoint,
    load_checkpoint,
)
from chessbot.training.dataset import ChessPositions, make_loader  # noqa: E402
from chessbot.training.loop import (  # noqa: E402
    TrainConfig,
    build_optimizer,
    build_scheduler,
    evaluate,
    train,
)
from chessbot.training.loss import LossWeights  # noqa: E402
from chessbot.utils.seed import set_seed  # noqa: E402

BOLD, GREY, RESET = ("\033[1m", "\033[90m", "\033[0m") if sys.stdout.isatty() else ("", "", "")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "processed")
    ap.add_argument("--run", default="supervised", help="nome della cartella in runs/")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup-steps", type=int, default=2000)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--max-steps", type=int, default=None, help="per gli smoke test")
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--checkpoint-every", type=int, default=2000)
    ap.add_argument("--no-amp", action="store_true", help="disattiva la precisione mista")
    ap.add_argument("--fresh", action="store_true", help="ignora i checkpoint e riparti da zero")
    ap.add_argument("--train-limit", type=int, default=None, help="usa solo N posizioni di train")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = ROOT / "runs" / args.run
    run_dir.mkdir(parents=True, exist_ok=True)

    config = TrainConfig(
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        amp=not args.no_amp,
        eval_every_steps=args.eval_every,
        checkpoint_every_steps=args.checkpoint_every,
        max_steps=args.max_steps,
    )

    print(f"{BOLD}Training supervisionato — Stadio 4{RESET}")
    print(f"  run: {run_dir}")
    print(f"  seed {args.seed}, batch {args.batch_size}, lr {args.lr}, {args.epochs} epoche")
    print()

    # --- dati -------------------------------------------------------------------------
    train_ds = ChessPositions(args.data, "train", limit=args.train_limit)
    val_ds = ChessPositions(args.data, "val")
    print(f"  train: {len(train_ds):,} posizioni")
    print(f"  val  : {len(val_ds):,} posizioni")

    train_loader = make_loader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    val_loader = make_loader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(2, args.num_workers // 2),
        seed=args.seed,
    )

    # --- rete -------------------------------------------------------------------------
    model = ChessNet(NetworkConfig(channels=args.channels, blocks=args.blocks)).to(device)
    print(f"  {model.describe()}")

    optimizer = build_optimizer(model, config)
    total_steps = config.max_steps or len(train_loader) * config.epochs
    scheduler = build_scheduler(optimizer, config, total_steps)
    use_amp = config.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    # --- ripresa ----------------------------------------------------------------------
    state = TrainingState()
    last = run_dir / "last.pt"
    if last.exists() and not args.fresh:
        print()
        print(f"{BOLD}Ripresa da un checkpoint esistente{RESET}")
        print("  " + inspect_checkpoint(last).replace("\n", "\n  "))
        state, _ = load_checkpoint(
            last,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            map_location=device,
        )
        print(f"  riparto dall'epoca {state.epoch}, step {state.global_step:,}")
    elif last.exists() and args.fresh:
        print(f"  {GREY}--fresh: ignoro il checkpoint esistente e riparto da zero{RESET}")

    print()
    try:
        state = train(
            model,
            train_loader,
            config=config,
            device=device,
            val_loader=val_loader,
            weights=LossWeights(),
            run_dir=run_dir,
            state=state,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )
    except KeyboardInterrupt:
        # Il salvataggio finale dentro `train` non viene raggiunto se si interrompe a
        # mano: qui si assicura che il lavoro fatto non vada perso comunque.
        from chessbot.training.checkpoint import CheckpointMeta, save_checkpoint

        print("\n  interrotto — salvo il checkpoint prima di uscire")
        save_checkpoint(
            run_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            state=state,
            meta=CheckpointMeta(notes="interrotto da tastiera"),
        )
        return 130

    # --- valutazione finale sull'intero set di validation -------------------------------
    print()
    print(f"{BOLD}Valutazione finale su tutto lo split val{RESET}")
    metrics = evaluate(model, val_loader, device, LossWeights())
    print(f"  loss     {metrics['loss']:.4f}")
    print(f"  top-1    {100 * metrics['top1']:.2f}%   {GREY}(Gate 4: 45-52%){RESET}")
    print(f"  top-5    {100 * metrics['top5']:.2f}%")
    print(f"  wdl acc  {100 * metrics['wdl_acc']:.2f}%")
    print()
    print(f"  {state.samples_seen:,} posizioni viste in {state.elapsed_seconds / 3600:.1f} h")
    print(f"  checkpoint: {run_dir / 'last.pt'}")
    print(f"  migliore  : {run_dir / 'best.pt'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
