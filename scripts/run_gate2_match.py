"""Match della baseline contro l'avversario casuale — criterio del Gate 2.

    python scripts/run_gate2_match.py --games 100 --depth 3

Il risultato viene stampato e salvato in `runs/gate2/`, cosi finisce in
docs/RESULTS.md con il commit hash accanto (regola #2: un risultato che non sai
riprodurre non e un risultato).

Sta in uno script e non in un test perche 100 partite costano minuti, non secondi: il
test automatico ne gioca poche (livello L1), il numero ufficiale si produce da qui.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from chessbot.baseline.match import play_match, random_player  # noqa: E402
from chessbot.baseline.search import best_move  # noqa: E402
from chessbot.utils.seed import set_seed  # noqa: E402


def commit_hash() -> str:
    p = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    return p.stdout.strip() or "sconosciuto"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "gate2")
    args = ap.parse_args()

    set_seed(args.seed)
    rng = random.Random(args.seed)

    print(f"Baseline (depth {args.depth}) vs random — {args.games} partite, seed {args.seed}")
    started = time.perf_counter()
    result = play_match(
        lambda b: best_move(b, args.depth, rng=rng),
        random_player(rng),
        games=args.games,
    )
    elapsed = time.perf_counter() - started

    print(f"  {result.summary()}")
    print(f"  {elapsed:.0f}s totali, {elapsed / args.games:.1f}s per partita")

    criterio = result.wins == args.games
    print(f"  criterio Gate 2 (vince tutte): {'PASS' if criterio else 'FAIL'}")

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "timestamp_utc": stamp,
        "commit": commit_hash(),
        "opponent": "random",
        "depth": args.depth,
        "seed": args.seed,
        "games": result.games,
        "wins": result.wins,
        "losses": result.losses,
        "draws": result.draws,
        "score_rate": result.score_rate,
        "summary": result.summary(),
        "elapsed_seconds": round(elapsed, 1),
        "criterio_gate2_vince_tutte": criterio,
    }
    path = args.out / f"match_random_{stamp}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"  salvato in {path.relative_to(ROOT)}")

    return 0 if criterio else 1


if __name__ == "__main__":
    sys.exit(main())
