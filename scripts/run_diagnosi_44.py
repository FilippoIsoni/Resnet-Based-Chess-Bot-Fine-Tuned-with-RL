"""Diagnosi §4.4: la value head regge la ricerca? (criterio del Gate 5).

    python scripts/run_diagnosi_44.py --games 200
    python scripts/run_diagnosi_44.py --games 40 --sims 400     # versione rapida

> Il rischio strutturale del piano (niente distillazione) si manifesta qui.
> **Misurarlo subito**, non a fine progetto.

Match fra **policy pura** e **policy + MCTS**. Il piano fissa la soglia:

> Se il guadagno e < 150 Elo, la value head non regge la ricerca.

## Perche questa misura e diversa dal test di monotonicita

La monotonicita (800 sim contro 50) dice se *piu ricerca* aiuta. Questa dice se *la
ricerca in se* aiuta, partendo da zero. Sono domande diverse: una ricerca puo migliorare
passando da 50 a 800 simulazioni e comunque non superare di molto la policy nuda, se il
valore che guida la ricerca e rumore.

## Come leggere il risultato

| Guadagno | Diagnosi | Azione |
|---|---|---|
| > 150 Elo | la value head regge | procedere alla Fase 5 |
| 50-150 Elo | debole ma utilizzabile | valutare le mitigazioni prima dell'RL |
| < 50 Elo | la ricerca e guidata da rumore | mitigare prima di procedere |

Mitigazioni previste dal piano, in ordine:

1. aumentare il peso della loss del valore da 0.5 a 1.0 e riallenare le ultime epoche
2. sfruttare tutte le partite con annotazioni `[%eval]` (ne abbiamo il 34%)
3. ridurre `c_puct` per fidarsi piu della policy e meno del valore
4. ultima risorsa: recuperare la distillazione

> **Regola #1.** Con 200 partite l'errore su una misura Elo e circa ±35. Un guadagno di
> 140 Elo non e distinguibile da uno di 170: se il risultato cade a cavallo della soglia,
> serve piu campione prima di decidere.
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

from chessbot.baseline.match import MatchResult, play_game  # noqa: E402
from chessbot.model.network import ChessNet, NetworkConfig  # noqa: E402
from chessbot.search import Evaluator, SearchConfig, run_mcts  # noqa: E402
from chessbot.training.checkpoint import commit_hash, load_checkpoint  # noqa: E402
from chessbot.utils.seed import set_seed  # noqa: E402

_COLOR = sys.stdout.isatty()
GREEN, RED, YELLOW, GREY, BOLD, RESET = (
    ("\033[32m", "\033[31m", "\033[33m", "\033[90m", "\033[1m", "\033[0m")
    if _COLOR
    else ("", "", "", "", "", "")
)

SOGLIA_ELO = 150.0


def mcts_player(evaluator, sims: int, seed: int):
    gen = np.random.default_rng(seed)

    def play(board: chess.Board) -> chess.Move | None:
        return run_mcts(board, evaluator, SearchConfig(simulations=sims), rng=gen).move

    return play


def policy_player(evaluator, rng: random.Random):
    """La policy della rete, campionata dalla sua distribuzione.

    Non l'argmax: sarebbe deterministica, e due partite con lo stesso colore sarebbero
    identiche. Campionare rende il match statisticamente significativo.
    """

    def play(board: chess.Board) -> chess.Move | None:
        priors, _ = evaluator.evaluate([board])[0]
        if not priors:
            return None
        moves = list(priors)
        weights = [priors[m] for m in moves]
        total = sum(weights)
        if total <= 0:
            return rng.choice(moves)
        return rng.choices(moves, weights=weights, k=1)[0]

    return play


def diagnose(result: MatchResult) -> tuple[str, list[str]]:
    """Traduce il risultato nella diagnosi e nell'azione previste da §4.4."""
    elo = result.elo_diff
    margin = result.elo_margin

    if elo == float("inf"):
        return "ok", [
            f"{GREEN}La value head regge.{RESET} Cappotto: la ricerca domina la policy pura.",
            "Si puo procedere alla Fase 5.",
        ]

    lines: list[str] = []
    if elo >= SOGLIA_ELO:
        stato = "ok"
        lines.append(f"{GREEN}La value head regge la ricerca.{RESET}")
        lines.append(f"Guadagno {elo:+.0f} Elo, sopra la soglia di {SOGLIA_ELO:.0f}.")
        lines.append("Si puo procedere alla Fase 5.")
    elif elo >= 50:
        stato = "debole"
        lines.append(f"{YELLOW}La value head e debole ma utilizzabile.{RESET}")
        lines.append(f"Guadagno {elo:+.0f} Elo, sotto la soglia di {SOGLIA_ELO:.0f}.")
        lines.append("Valutare le mitigazioni §4.4 prima dell'RL.")
    else:
        stato = "insufficiente"
        lines.append(f"{RED}La ricerca e guidata da rumore.{RESET}")
        lines.append(f"Guadagno {elo:+.0f} Elo: l'MCTS aggiunge poco o nulla.")
        lines.append("Applicare le mitigazioni PRIMA di procedere.")

    # Regola #1: se l'intervallo di confidenza scavalca la soglia, il verdetto e incerto.
    if abs(elo - SOGLIA_ELO) < margin:
        lines.append(
            f"{YELLOW}Attenzione:{RESET} l'intervallo (±{margin:.0f}) scavalca la soglia. "
            "Il verdetto non e statisticamente solido — servono piu partite."
        )

    if stato != "ok":
        lines.append("")
        lines.append("Mitigazioni previste da §4.4, in ordine:")
        lines.append("  1. peso della loss del valore da 0.5 a 1.0, riallenare le ultime epoche")
        lines.append("  2. sfruttare tutte le partite con [%eval] (ne abbiamo il 34%)")
        lines.append("  3. ridurre c_puct: piu fiducia nella policy, meno nel valore")
        lines.append("  4. ultima risorsa: recuperare la distillazione da Stockfish")

    return stato, lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "runs" / "supervised" / "best.pt")
    ap.add_argument("--games", type=int, default=200, help="§4.4 ne prevede 200")
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--max-plies", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "gate5")
    args = ap.parse_args()

    if not args.checkpoint.exists():
        print(f"checkpoint assente: {args.checkpoint}")
        return 1

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"{BOLD}Diagnosi §4.4 — la value head regge la ricerca?{RESET}")
    print(f"  policy + MCTS {args.sims} simulazioni  vs  policy pura")
    print(f"  {args.games} partite, colori alternati, seed {args.seed}")
    print(f"  soglia del piano: {SOGLIA_ELO:.0f} Elo")
    print()

    net = ChessNet(NetworkConfig()).to(device)
    load_checkpoint(args.checkpoint, model=net, map_location=device)
    evaluator = Evaluator(net, device)

    rng = random.Random(args.seed)
    mcts = mcts_player(evaluator, args.sims, args.seed)
    policy = policy_player(evaluator, rng)

    result = MatchResult()
    started = time.perf_counter()
    last_report = 0.0

    for i in range(args.games):
        mcts_is_white = i % 2 == 0
        white, black = (mcts, policy) if mcts_is_white else (policy, mcts)
        outcome, _ = play_game(white, black, max_plies=args.max_plies)

        if outcome == "1/2-1/2":
            result.draws += 1
        elif (outcome == "1-0") == mcts_is_white:
            result.wins += 1
        else:
            result.losses += 1

        now = time.perf_counter()
        if now - last_report >= 30:
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
    print(f"  {GREY}{elapsed / 60:.1f} min, {elapsed / args.games:.1f} s per partita{RESET}")
    print()
    print(f"{BOLD}Diagnosi{RESET}")
    stato, lines = diagnose(result)
    for line in lines:
        print(f"  {line}")

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "timestamp_utc": stamp,
        "commit": commit_hash(),
        "checkpoint": str(args.checkpoint.name),
        "simulations": args.sims,
        "games": result.games,
        "wins": result.wins,
        "losses": result.losses,
        "draws": result.draws,
        "score_rate": result.score_rate,
        "elo_diff": None if result.elo_diff in (float("inf"), float("-inf")) else result.elo_diff,
        "elo_margin": None if result.elo_margin == float("inf") else result.elo_margin,
        "summary": result.summary(),
        "diagnosi": stato,
        "soglia_elo": SOGLIA_ELO,
        "elapsed_seconds": round(elapsed, 1),
    }
    path = args.out / f"diagnosi_44_{stamp}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print()
    print(f"  salvato in {path.relative_to(ROOT)}")

    return 0 if stato == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
