"""Scala Elo contro Stockfish — la misura assoluta della forza (Appendice A del piano).

    python scripts/run_elo_ladder.py
    python scripts/run_elo_ladder.py --levels 1400 1800 2200 --games 60

> Stockfish con `UCI_LimitStrength=true` e `UCI_Elo` crescente: **1400 -> 1800 -> 2200**.
> Metro calibrato e gratuito.

## Perche serve, dato che abbiamo gia tre misure

Le misure fatte finora sono tutte **relative**: baseline contro random, MCTS contro
policy pura, RL contro supervisionata. Dicono di quanto e migliorato il motore rispetto a
se stesso, non **quanto e forte**.

Per un numero assoluto serve un avversario di forza nota, e Stockfish con la forza
limitata e esattamente questo.

## Come si stima l'Elo dalla scala

Contro un avversario di Elo noto `E`, un punteggio `s` corrisponde a:

    Elo_nostro = E - 400 * log10(1/s - 1)

Ogni livello da una stima indipendente. Se sono coerenti fra loro, la misura e
affidabile; se divergono molto, significa che il nostro motore ha un profilo di forza
irregolare — forte in tattica e debole in posizionale, per esempio — e un solo numero
non lo descrive bene.

> **Attenzione al soffitto e al pavimento.** Se il motore vince 100/100 contro un
> livello, quel livello dice solo "siamo sopra": la stima e infinita e va scartata. Lo
> stesso per 0/100. Solo i livelli con un punteggio intermedio informano davvero.

## Nota sul limite di Stockfish

`UCI_Elo` accetta da 1320 in su su questa build (Stockfish 18), non da 1400 come il piano
assume. Il primo gradino usa quindi 1400 se disponibile, altrimenti il minimo.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import chess  # noqa: E402
import chess.engine  # noqa: E402
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

STOCKFISH = ROOT / "tools" / "stockfish" / "stockfish.exe"


def elo_from_score(opponent_elo: float, score_rate: float, games: int) -> tuple[float, float]:
    """Elo stimato e semiampiezza dell'IC al 95%, dal punteggio contro un avversario noto.

    Restituisce +-inf sui risultati estremi: con 100% o 0% il modello logistico non ha
    soluzione finita, e inventare un numero sarebbe peggio che dire "non stimabile"
    (regola #1).
    """
    if score_rate <= 0.0:
        return float("-inf"), float("inf")
    if score_rate >= 1.0:
        return float("inf"), float("inf")

    diff = -400.0 * math.log10(1.0 / score_rate - 1.0)
    elo = opponent_elo + diff

    se = math.sqrt(score_rate * (1.0 - score_rate) / games)
    derivative = 400.0 / (math.log(10.0) * score_rate * (1.0 - score_rate))
    return elo, 1.96 * se * derivative


def play_level(
    evaluator,
    engine: chess.engine.SimpleEngine,
    level_elo: int,
    *,
    games: int,
    sims: int,
    move_time: float,
    openings: list[chess.Board],
    max_plies: int,
    rng: np.random.Generator,
) -> MatchResult:
    """Un match contro Stockfish limitato a `level_elo`."""
    engine.configure({"UCI_LimitStrength": True, "UCI_Elo": level_elo})
    limit = chess.engine.Limit(time=move_time)

    result = MatchResult()
    started = time.perf_counter()
    last_report = 0.0

    for i in range(games):
        board = sample_openings(openings, 1, rng)[0] if openings else chess.Board()
        we_are_white = i % 2 == 0

        while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
            if (board.turn == chess.WHITE) == we_are_white:
                move = run_mcts(board, evaluator, SearchConfig(simulations=sims), rng=rng).move
            else:
                move = engine.play(board, limit).move
            if move is None or move not in board.legal_moves:
                break
            board.push(move)

        outcome = (
            board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else "1/2-1/2"
        )
        if outcome == "1/2-1/2":
            result.draws += 1
        elif (outcome == "1-0") == we_are_white:
            result.wins += 1
        else:
            result.losses += 1

        now = time.perf_counter()
        if now - last_report >= 60:
            print(
                f"    {i + 1}/{games}: {result.wins}W-{result.losses}L-{result.draws}D "
                f"({100 * result.score_rate:.1f}%) — {(now - started) / 60:.0f} min",
                flush=True,
            )
            last_report = now

    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "runs" / "rl" / "main" / "state.pt")
    ap.add_argument("--levels", type=int, nargs="+", default=[1400, 1800, 2200])
    ap.add_argument("--games", type=int, default=60, help="partite per livello")
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--move-time", type=float, default=0.1, help="secondi per mossa a Stockfish")
    ap.add_argument("--max-plies", type=int, default=200)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "elo")
    args = ap.parse_args()

    if not STOCKFISH.exists():
        print(f"Stockfish assente: {STOCKFISH}")
        return 1
    if not args.checkpoint.exists():
        print(f"checkpoint assente: {args.checkpoint}")
        return 1

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"{BOLD}Scala Elo contro Stockfish{RESET}")
    print(f"  livelli: {', '.join(str(x) for x in args.levels)}")
    print(f"  {args.games} partite per livello, {args.sims} simulazioni")
    print(f"  Stockfish: {args.move_time}s per mossa")
    print()

    # Il checkpoint RL ha un formato diverso da quelli supervisionati.
    net = ChessNet(NetworkConfig()).to(device)
    if args.checkpoint.name == "state.pt":
        payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
        net.load_state_dict(payload["champion"])
        print(f"  motore: RL iterazione {payload['iteration']}")
    else:
        load_checkpoint(args.checkpoint, model=net, map_location=device)
        print(f"  motore: {args.checkpoint.name}")
    net.eval()
    evaluator = Evaluator(net, device)

    book_path = ROOT / "data" / "books" / "openings.json"
    openings = load_book(book_path) if book_path.exists() else []
    rng = np.random.default_rng(args.seed)

    engine = chess.engine.SimpleEngine.popen_uci(str(STOCKFISH.resolve()))
    misure: list[dict] = []

    try:
        for level in args.levels:
            print(f"{BOLD}  vs Stockfish {level}{RESET}")
            started = time.perf_counter()
            result = play_level(
                evaluator,
                engine,
                level,
                games=args.games,
                sims=args.sims,
                move_time=args.move_time,
                openings=openings,
                max_plies=args.max_plies,
                rng=rng,
            )
            elapsed = time.perf_counter() - started
            elo, margin = elo_from_score(level, result.score_rate, result.games)

            estrema = not math.isfinite(elo)
            testo = (
                "oltre la scala (nessuna sconfitta)"
                if elo == float("inf")
                else "sotto la scala (nessuna vittoria)"
                if elo == float("-inf")
                else f"{elo:.0f} ± {margin:.0f}"
            )
            print(f"    {result.summary()}")
            print(f"    Elo stimato: {BOLD}{testo}{RESET}  {GREY}({elapsed / 60:.0f} min){RESET}")
            print()

            misure.append(
                {
                    "opponent_elo": level,
                    "games": result.games,
                    "wins": result.wins,
                    "losses": result.losses,
                    "draws": result.draws,
                    "score_rate": result.score_rate,
                    "elo_estimate": None if estrema else round(elo, 1),
                    "elo_margin": None if estrema else round(margin, 1),
                    "usable": not estrema,
                    "elapsed_seconds": round(elapsed, 1),
                }
            )
    finally:
        engine.quit()

    # --- stima complessiva --------------------------------------------------------------
    usabili = [m for m in misure if m["usable"]]

    print(f"{BOLD}Stima complessiva{RESET}")
    if not usabili:
        print(f"  {YELLOW}Tutti i livelli hanno dato risultati estremi.{RESET}")
        print("  Il motore e fuori scala: rifare con livelli piu adatti.")
        stima = None
    else:
        # Media pesata sull'inverso della varianza: i livelli con IC piu stretto contano
        # di piu. E il modo corretto di combinare misure indipendenti della stessa
        # quantita — al contrario della somma, che era l'errore dello Stadio 6.
        pesi = [1.0 / (m["elo_margin"] ** 2) for m in usabili]
        stima = sum(m["elo_estimate"] * w for m, w in zip(usabili, pesi, strict=True)) / sum(pesi)
        margine = math.sqrt(1.0 / sum(pesi))

        for m in usabili:
            print(
                f"  vs {m['opponent_elo']}: {100 * m['score_rate']:5.1f}% -> "
                f"{m['elo_estimate']:.0f} ± {m['elo_margin']:.0f}"
            )
        print()
        print(f"  {BOLD}Elo stimato: {stima:.0f} ± {margine:.0f}{RESET}")

        spread = max(m["elo_estimate"] for m in usabili) - min(m["elo_estimate"] for m in usabili)
        if spread > 200:
            print(f"  {YELLOW}Le stime dei singoli livelli divergono di {spread:.0f} Elo:{RESET}")
            print("  il profilo di forza e irregolare e un numero solo lo descrive male.")

    scartati = [m for m in misure if not m["usable"]]
    for m in scartati:
        print(
            f"  {GREY}vs {m['opponent_elo']}: {100 * m['score_rate']:.0f}% — "
            f"risultato estremo, scartato dalla stima{RESET}"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    payload_out = {
        "timestamp_utc": stamp,
        "commit": commit_hash(),
        "checkpoint": str(args.checkpoint.name),
        "simulations": args.sims,
        "stockfish_move_time": args.move_time,
        "levels": misure,
        "elo_estimate": None if stima is None else round(stima, 1),
        "elo_margin": None if stima is None else round(margine, 1),
    }
    path = args.out / f"ladder_{stamp}.json"
    path.write_text(json.dumps(payload_out, indent=2) + "\n", encoding="utf-8")
    print()
    print(f"  salvato in {path.relative_to(ROOT)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
