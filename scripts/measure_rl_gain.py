"""Misura il guadagno vero dell'RL: rete finale contro quella supervisionata.

    python scripts/measure_rl_gain.py --games 200

## Perche non basta l'Elo cumulativo del ciclo

`summary.json` riporta la somma delle stime dei singoli gating. Quel numero **non e una
misura**, per due ragioni:

1. **Gli errori si sommano.** Ogni gating ha un IC di circa ±88 Elo su 60 partite;
   sommandone sette l'incertezza complessiva sale a ~±233. Un "+380 ± 233" non dice
   granche.
2. **Misura la cosa sbagliata.** Ogni gating confronta la rete nuova con la
   **precedente**, non con quella di partenza. I guadagni relativi non si sommano
   linearmente: se A batte B di 40 Elo e B batte C di 40, A non batte necessariamente C
   di 80.

L'unico numero difendibile e il confronto diretto **finale contro iniziale**, su un
campione abbastanza grande da avere un intervallo di confidenza stretto.

> Regola #1: ogni numero riportato ha un intervallo di confidenza. Con 200 partite
> l'errore e circa ±35 Elo.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import chess  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from chessbot.baseline.match import MatchResult  # noqa: E402
from chessbot.model.network import ChessNet, NetworkConfig  # noqa: E402
from chessbot.search import Evaluator, SearchConfig, run_mcts  # noqa: E402
from chessbot.search.openings import load_book, sample_openings  # noqa: E402
from chessbot.training.checkpoint import commit_hash, load_checkpoint  # noqa: E402
from chessbot.utils.seed import set_seed  # noqa: E402

GREEN, RED, YELLOW, GREY, BOLD, RESET = (
    ("\033[32m", "\033[31m", "\033[33m", "\033[90m", "\033[1m", "\033[0m")
    if sys.stdout.isatty()
    else ("", "", "", "", "", "")
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rl", type=Path, default=ROOT / "runs" / "rl" / "main" / "state.pt")
    ap.add_argument("--supervised", type=Path, default=ROOT / "runs" / "supervised" / "best.pt")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--max-plies", type=int, default=200)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "gate6")
    args = ap.parse_args()

    for path in (args.rl, args.supervised):
        if not path.exists():
            print(f"checkpoint assente: {path}")
            return 1

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"{BOLD}Guadagno dell'RL: rete finale vs supervisionata{RESET}")
    print(f"  {args.games} partite, {args.sims} simulazioni, colori alternati")
    print()

    rl_net = ChessNet(NetworkConfig()).to(device)
    payload = torch.load(args.rl, map_location=device, weights_only=False)
    rl_net.load_state_dict(payload["champion"])
    rl_net.eval()
    print(f"  RL          : iterazione {payload['iteration']}, commit {payload.get('commit')}")

    sup_net = ChessNet(NetworkConfig()).to(device)
    load_checkpoint(args.supervised, model=sup_net, map_location=device)
    sup_net.eval()
    print(f"  supervisionata: {args.supervised.name}")
    print()

    rl_eval = Evaluator(rl_net, device)
    sup_eval = Evaluator(sup_net, device)
    search = SearchConfig(simulations=args.sims)

    book_path = ROOT / "data" / "books" / "openings.json"
    openings = load_book(book_path) if book_path.exists() else []
    rng = np.random.default_rng(args.seed)
    py_rng = random.Random(args.seed)

    result = MatchResult()
    started = time.perf_counter()
    last_report = 0.0

    for i in range(args.games):
        board = sample_openings(openings, 1, rng)[0] if openings else chess.Board()
        rl_is_white = i % 2 == 0

        while not board.is_game_over(claim_draw=True) and board.ply() < args.max_plies:
            is_rl = (board.turn == chess.WHITE) == rl_is_white
            search_result = run_mcts(board, rl_eval if is_rl else sup_eval, search, rng=rng)
            if search_result.move is None:
                break
            board.push(search_result.move)

        outcome = (
            board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else "1/2-1/2"
        )
        if outcome == "1/2-1/2":
            result.draws += 1
        elif (outcome == "1-0") == rl_is_white:
            result.wins += 1
        else:
            result.losses += 1

        now = time.perf_counter()
        if now - last_report >= 60:
            print(
                f"  {i + 1}/{args.games}: {result.wins}W-{result.losses}L-{result.draws}D "
                f"({100 * result.score_rate:.1f}%) — {(now - started) / 60:.0f} min",
                flush=True,
            )
            last_report = now

    elapsed = time.perf_counter() - started

    print()
    print(f"{BOLD}Risultato{RESET}")
    print(f"  {result.summary()}")
    print(f"  {GREY}{elapsed / 60:.1f} min{RESET}")
    print()

    elo = result.elo_diff
    margin = result.elo_margin
    print(f"{BOLD}Verdetto{RESET}")
    if not np.isfinite(elo):
        print(f"  {GREEN}Risultato estremo: l'RL domina.{RESET}")
        verdict = "estremo"
    elif elo - margin > 0:
        print(f"  {GREEN}L'RL ha migliorato la rete: {elo:+.0f} ± {margin:.0f} Elo.{RESET}")
        print("  L'intervallo di confidenza sta tutto sopra lo zero.")
        verdict = "migliorata"
    elif elo > 0:
        print(f"  {YELLOW}Miglioramento non dimostrato: {elo:+.0f} ± {margin:.0f} Elo.{RESET}")
        print("  L'intervallo contiene lo zero: il guadagno non e distinguibile dal rumore.")
        verdict = "incerto"
    else:
        print(f"  {RED}Nessun miglioramento: {elo:+.0f} ± {margin:.0f} Elo.{RESET}")
        verdict = "nessuno"

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    payload_out = {
        "timestamp_utc": stamp,
        "commit": commit_hash(),
        "rl_checkpoint": str(args.rl.name),
        "rl_iteration": payload["iteration"],
        "supervised_checkpoint": str(args.supervised.name),
        "simulations": args.sims,
        "games": result.games,
        "wins": result.wins,
        "losses": result.losses,
        "draws": result.draws,
        "score_rate": result.score_rate,
        "elo_diff": None if not np.isfinite(elo) else elo,
        "elo_margin": None if not np.isfinite(margin) else margin,
        "verdict": verdict,
        "summary": result.summary(),
        "elapsed_seconds": round(elapsed, 1),
    }
    path = args.out / f"rl_gain_{stamp}.json"
    path.write_text(json.dumps(payload_out, indent=2) + "\n", encoding="utf-8")
    print()
    print(f"  salvato in {path.relative_to(ROOT)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
