"""Misura lo speedup del batching fra partite — criterio del Gate 6 (criticita #8).

    python scripts/measure_speedup.py

> Giocare 64-128 partite in parallelo nello stesso processo, batchando le valutazioni di
> rete **fra partite diverse**, non solo fra foglie dello stesso albero.
> Guadagno stimato: 5-10x. Da implementare *prima* di lanciare la prima iterazione.

Confronta le partite/ora con una partita alla volta contro N in parallelo, e salva il
risultato in `runs/gate6/speedup.json` perche il gate possa leggerlo senza rifare la
misura.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch  # noqa: E402

from chessbot.model.network import ChessNet, NetworkConfig  # noqa: E402
from chessbot.search import Evaluator, SearchConfig  # noqa: E402
from chessbot.search.parallel import self_play_batch  # noqa: E402
from chessbot.training.checkpoint import commit_hash, load_checkpoint  # noqa: E402
from chessbot.utils.seed import set_seed  # noqa: E402

GREEN, RED, GREY, BOLD, RESET = (
    ("\033[32m", "\033[31m", "\033[90m", "\033[1m", "\033[0m")
    if sys.stdout.isatty()
    else ("", "", "", "", "")
)

SOGLIA = 5.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "runs" / "supervised" / "best.pt")
    ap.add_argument("--sims", type=int, default=50)
    ap.add_argument("--max-plies", type=int, default=40)
    ap.add_argument("--parallel", type=int, nargs="+", default=[1, 8, 32, 64])
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "gate6" / "speedup.json")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    if not args.checkpoint.exists():
        print(f"checkpoint assente: {args.checkpoint}")
        return 1

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = ChessNet(NetworkConfig()).to(device)
    load_checkpoint(args.checkpoint, model=net, map_location=device)
    evaluator = Evaluator(net, device)
    config = SearchConfig(simulations=args.sims, dirichlet_epsilon=0.25)

    print(f"{BOLD}Speedup del batching fra partite (criticita #8){RESET}")
    print(f"  {args.sims} simulazioni/mossa, max {args.max_plies} semimosse, device {device}")
    print()

    misure: list[dict] = []
    print(f"  {'partite':>8} {'tempo':>8} {'partite/h':>11} {'batch':>7}")
    print("  " + "-" * 38)

    for n in args.parallel:
        _records, stats = self_play_batch(
            evaluator, n_games=n, config=config, max_plies=args.max_plies
        )
        misure.append(
            {
                "parallel_games": n,
                "seconds": round(stats.elapsed, 2),
                "games_per_hour": round(stats.games_per_hour, 1),
                "avg_batch": round(stats.avg_batch, 1),
                "positions": stats.positions,
            }
        )
        print(
            f"  {n:>8} {stats.elapsed:>7.1f}s {stats.games_per_hour:>11.0f} "
            f"{stats.avg_batch:>7.1f}",
            flush=True,
        )

    baseline = misure[0]["games_per_hour"]
    best = max(m["games_per_hour"] for m in misure)
    speedup = best / baseline if baseline else 0.0
    best_n = next(m["parallel_games"] for m in misure if m["games_per_hour"] == best)

    print()
    ok = speedup >= SOGLIA
    colore = GREEN if ok else RED
    print(f"  speedup: {colore}{speedup:.1f}x{RESET} (soglia del Gate 6: {SOGLIA:.0f}x)")
    print(f"  ottimale a {best_n} partite in parallelo")
    if not ok:
        print(f"  {RED}sotto la soglia: rivedere il batching prima di lanciare l'RL{RESET}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commit": commit_hash(),
        "simulations": args.sims,
        "measurements": misure,
        "baseline_games_per_hour": baseline,
        "best_games_per_hour": best,
        "best_parallel_games": best_n,
        "speedup": round(speedup, 2),
        "threshold": SOGLIA,
        "passed": ok,
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"  {GREY}salvato in {args.out.relative_to(ROOT)}{RESET}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
