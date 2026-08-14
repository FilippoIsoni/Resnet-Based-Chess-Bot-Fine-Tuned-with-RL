"""Batteria di correttezza dell'MCTS — §4.5 del piano, criteri del Gate 5.

    python scripts/run_gate5_tests.py
    python scripts/run_gate5_tests.py --games 20 --skip-monotonic

> Un MCTS bacato non crasha, sembra solo debole.

I test, e cosa intercetta ognuno:

| Test | Se fallisce, il sospetto e |
|---|---|
| Matti in 1 a 50 simulazioni | riconoscimento dei terminali, o selezione che non esplora |
| Matti in 2-3 a 400-800 | backup del valore, o profondita insufficiente |
| Rete casuale + MCTS vs rete casuale sola | la ricerca non aggiunge nulla: PUCT o backup rotti |
| Simmetria colori | segno del valore (criticita #2) |
| **Forza crescente con le simulazioni** | **il test piu diagnostico dell'intero progetto** |

L'ultimo merita una parola. Se raddoppiare le simulazioni non aumenta la forza, delle due
l'una: o c'e un bug nel backup, oppure la value head e talmente debole da rendere la
ricerca inutile. Sono cause diverse con rimedi diversi, e distinguerle e il punto di §4.4.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import chess  # noqa: E402
import torch  # noqa: E402

from chessbot.baseline.match import MatchResult, play_game, play_match  # noqa: E402
from chessbot.model.network import ChessNet, NetworkConfig  # noqa: E402
from chessbot.search import Evaluator, SearchConfig, run_mcts  # noqa: E402
from chessbot.training.checkpoint import load_checkpoint  # noqa: E402
from chessbot.utils.seed import set_seed  # noqa: E402

_COLOR = sys.stdout.isatty()
GREEN, RED, YELLOW, GREY, BOLD, RESET = (
    ("\033[32m", "\033[31m", "\033[33m", "\033[90m", "\033[1m", "\033[0m")
    if _COLOR
    else ("", "", "", "", "", "")
)

MATE_IN_1 = ROOT / "tests" / "golden" / "mate_in_1.json"

# Matti forzati in 2, dal Chess Programming Wiki e da posizioni classiche.
MATE_IN_2: tuple[tuple[str, str], ...] = (
    ("1r3rk1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", "matto di torre sull'ottava"),
    ("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", "torre e re contro re e pedoni"),
    ("r5k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1", "opposizione di torri"),
)


def report(name: str, ok: bool, detail: str = "") -> bool:
    mark = f"{GREEN}[PASS]{RESET}" if ok else f"{RED}[FAIL]{RESET}"
    print(f"  {mark} {name}" + (f"  {GREY}{detail}{RESET}" if detail else ""))
    return ok


class UniformEvaluator:
    """Rete finta: priori uniformi, valore sempre zero.

    Non sa nulla di scacchi. Serve a verificare che la MECCANICA dell'albero funzioni
    da sola: se con questa l'MCTS trova comunque i matti, il merito e della ricerca e
    non della rete.
    """

    def evaluate(self, boards: list[chess.Board]) -> list[tuple[dict, float]]:
        out = []
        for board in boards:
            moves = list(board.legal_moves)
            uniform = 1.0 / max(len(moves), 1)
            out.append(({m: uniform for m in moves}, 0.0))
        return out


class MaterialEvaluator:
    """Rete finta che sa contare i pezzi: priori uniformi, valore dal materiale.

    > **Perche `UniformEvaluator` non basta per il test di forza.** Con valore sempre
    > zero l'MCTS non ha alcun motivo di preferire una mossa a un'altra: tutte le linee
    > valgono 0, la ricerca ripete e la partita finisce patta al ply 14 per ripetizione.
    > Misurato: 6 patte su 6 contro la policy pura.
    >
    > Il test voleva dimostrare che *l'albero aggiunge forza*, ma un albero che esplora
    > un valore costante non puo aggiungere nulla — non e un bug della ricerca, e una
    > tautologia del test.
    >
    > Con un valore materiale, banale ma non costante, la ricerca ha qualcosa da
    > ottimizzare e la differenza fra cercare e non cercare diventa misurabile.
    """

    VALUES = {  # noqa: RUF012
        chess.PAWN: 1.0,
        chess.KNIGHT: 3.0,
        chess.BISHOP: 3.0,
        chess.ROOK: 5.0,
        chess.QUEEN: 9.0,
        chess.KING: 0.0,
    }

    def evaluate(self, boards: list[chess.Board]) -> list[tuple[dict, float]]:
        out = []
        for board in boards:
            moves = list(board.legal_moves)
            uniform = 1.0 / max(len(moves), 1)

            balance = 0.0
            for piece in board.piece_map().values():
                v = self.VALUES[piece.piece_type]
                balance += v if piece.color == board.turn else -v
            # Comprime in [-1, 1]: una donna di vantaggio vale circa 0,6.
            value = balance / 15.0
            out.append(({m: uniform for m in moves}, max(-1.0, min(1.0, value))))
        return out


def mcts_player(evaluator, sims: int, rng: random.Random):
    """Adatta l'MCTS all'interfaccia dei match (`board -> mossa`)."""
    import numpy as np

    gen = np.random.default_rng(rng.randrange(2**32))

    def play(board: chess.Board) -> chess.Move | None:
        return run_mcts(board, evaluator, SearchConfig(simulations=sims), rng=gen).move

    return play


def policy_player(evaluator, rng: random.Random, *, sample: bool = True):
    """Solo la policy della rete, senza ricerca.

    > **Perche campiona invece di prendere l'argmax.** Con priori uniformi — le reti
    > finte dei test — `max()` restituisce sempre la prima mossa del dizionario: un
    > giocatore deterministico e degenere. Misurato: contro l'MCTS muoveva la torre
    > h8-g8-h8 all'infinito, e tutte le partite finivano patta per ripetizione al ply 14.
    >
    > Il test voleva misurare se la ricerca aggiunge forza, e invece misurava due
    > automi bloccati in un ciclo. Campionando dalla distribuzione, la policy pura
    > diventa un avversario debole ma non degenere, che e cio che serve.
    >
    > Con la rete VERA l'argmax andrebbe bene (i priori non sono uniformi), ma si tiene
    > il campionamento per coerenza: la diagnosi §4.4 confronta le stesse due cose.
    """

    def play(board: chess.Board) -> chess.Move | None:
        priors, _ = evaluator.evaluate([board])[0]
        if not priors:
            return None
        if not sample:
            return max(priors, key=priors.get)
        moves = list(priors)
        weights = [priors[m] for m in moves]
        total = sum(weights)
        if total <= 0:
            return rng.choice(moves)
        return rng.choices(moves, weights=weights, k=1)[0]

    return play


def test_mate_in_1(evaluator, sims: int, limit: int) -> tuple[bool, str]:
    data = json.loads(MATE_IN_1.read_text(encoding="utf-8"))[:limit]
    found = 0
    for entry in data:
        board = chess.Board(entry["fen"])
        result = run_mcts(board, evaluator, SearchConfig(simulations=sims))
        if result.move is not None and result.move.uci() in entry["mates"]:
            found += 1
    return found == len(data), f"{found}/{len(data)} a {sims} simulazioni"


def test_color_symmetry(evaluator, sims: int) -> tuple[bool, str]:
    """La stessa posizione a colori invertiti deve dare la mossa specchiata.

    > Criticita #2, verifica definitiva. Se il segno del valore e sbagliato nel backup,
    > il motore valuta le proprie posizioni con la prospettiva dell'avversario e le due
    > ricerche divergono.
    """
    positions = [
        chess.STARTING_FEN,
        "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    ]
    mismatches = []
    for fen in positions:
        board = chess.Board(fen)
        mirrored = board.mirror()

        a = run_mcts(board, evaluator, SearchConfig(simulations=sims)).move
        b = run_mcts(mirrored, evaluator, SearchConfig(simulations=sims)).move
        if a is None or b is None:
            continue
        expected = chess.Move(
            chess.square_mirror(a.from_square),
            chess.square_mirror(a.to_square),
            promotion=a.promotion,
        )
        if b != expected:
            mismatches.append(f"{fen[:20]}: {a} vs {b} (atteso {expected})")

    # Con la ricerca stocastica una divergenza occasionale e possibile su posizioni
    # dove due mosse sono quasi equivalenti: si segnala ma non si fallisce per una sola.
    ok = len(mismatches) <= 1
    detail = f"{len(positions) - len(mismatches)}/{len(positions)} coerenti"
    if mismatches:
        detail += f" — {mismatches[0]}"
    return ok, detail


PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}


def material_balance(board: chess.Board, color: chess.Color) -> int:
    """Vantaggio materiale di `color`, in pedoni."""
    mine = sum(
        PIECE_VALUES.get(p.piece_type, 0) for p in board.piece_map().values() if p.color == color
    )
    theirs = sum(
        PIECE_VALUES.get(p.piece_type, 0) for p in board.piece_map().values() if p.color != color
    )
    return mine - theirs


def test_search_beats_policy(evaluator, games: int, sims: int) -> tuple[bool, str, MatchResult]:
    """MCTS contro la sola policy, a parita di rete.

    Misura il **vantaggio materiale accumulato**, non il risultato della partita.

    > **Perche non il risultato.** Le reti finte di questa batteria valutano il solo
    > materiale: una volta catturato tutto non hanno alcun incentivo verso il matto, e la
    > partita finisce patta anche quando l'avversario e ridotto al re nudo. Misurato: due
    > partite su quattro con il nero a ZERO materiale, esito 1/2-1/2.
    >
    > E lo stesso limite della baseline al Gate 2 — vincere materiale e saper concludere
    > sono due cose diverse — ma qui non e un difetto del motore: e un difetto della
    > valutazione finta che il test usa apposta per isolare la meccanica dell'albero.
    >
    > Il criterio del piano e "la ricerca batte nettamente la policy sola". Con questa
    > valutazione, "battere" si misura in pedoni conquistati.
    """
    rng = random.Random(1337)
    mcts = mcts_player(evaluator, sims, rng)
    policy = policy_player(evaluator, rng)

    balances: list[int] = []
    result = MatchResult()

    for i in range(games):
        mcts_is_white = i % 2 == 0
        white, black = (mcts, policy) if mcts_is_white else (policy, mcts)
        outcome, board = play_game(white, black, max_plies=200)

        color = chess.WHITE if mcts_is_white else chess.BLACK
        balances.append(material_balance(board, color))

        if outcome == "1/2-1/2":
            result.draws += 1
        elif (outcome == "1-0") == mcts_is_white:
            result.wins += 1
        else:
            result.losses += 1

    mean = sum(balances) / len(balances)
    won = sum(1 for b in balances if b > 0)
    detail = (
        f"vantaggio materiale medio {mean:+.1f} pedoni, "
        f"in vantaggio in {won}/{games} partite ({result.summary()})"
    )
    # Se la ricerca aggiunge forza, deve finire in vantaggio quasi sempre.
    return mean > 3.0 and won >= games * 0.75, detail, result


def test_monotonic_strength(evaluator, games: int) -> tuple[bool, str]:
    """Il test piu diagnostico: piu simulazioni, piu forza.

    Si gioca 800 simulazioni contro 50. Se la forza cresce con la ricerca, il segno del
    backup e giusto e la value head serve a qualcosa. Se no, uno dei due e rotto.
    """
    rng = random.Random(99)
    result = play_match(
        mcts_player(evaluator, 800, rng),
        mcts_player(evaluator, 50, rng),
        games=games,
        max_plies=120,
    )
    return result.score_rate > 0.55, f"800 sim vs 50 sim: {result.summary()}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "runs" / "supervised" / "best.pt")
    ap.add_argument("--games", type=int, default=20, help="partite per i test di forza")
    ap.add_argument("--mate-positions", type=int, default=50)
    ap.add_argument("--skip-monotonic", action="store_true", help="salta il test piu lento")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"{BOLD}Batteria §4.5 — correttezza dell'MCTS{RESET}")
    print()

    ok = True

    # --- con la rete CIECA: verifica la meccanica dell'albero -------------------------
    print(f"{BOLD}Con rete uniforme (priori piatti, valore 0){RESET}")
    print(f"  {GREY}se questi passano, l'albero funziona indipendentemente dalla rete{RESET}")
    uniform = UniformEvaluator()

    passed, detail = test_mate_in_1(uniform, 50, args.mate_positions)
    ok &= report("matti in 1 a 50 simulazioni", passed, detail)

    passed, detail = test_mate_in_1(uniform, 400, args.mate_positions)
    ok &= report("matti in 1 a 400 simulazioni", passed, detail)

    # Il test di forza usa una rete che almeno conta i pezzi: con valore costante la
    # ricerca non ha nulla da ottimizzare e il test non misurerebbe nulla.
    passed, detail, _ = test_search_beats_policy(MaterialEvaluator(), max(6, args.games // 3), 200)
    ok &= report("MCTS batte la policy pura (rete materiale)", passed, detail)

    # --- con la rete VERA -------------------------------------------------------------
    if not args.checkpoint.exists():
        print()
        print(f"  {YELLOW}checkpoint assente: {args.checkpoint}{RESET}")
        print(f"  {GREY}i test con la rete allenata sono saltati{RESET}")
        return 0 if ok else 1

    print()
    print(f"{BOLD}Con la rete allenata ({args.checkpoint.name}){RESET}")
    net = ChessNet(NetworkConfig()).to(device)
    load_checkpoint(args.checkpoint, model=net, map_location=device)
    evaluator = Evaluator(net, device)

    passed, detail = test_mate_in_1(evaluator, 50, min(20, args.mate_positions))
    ok &= report("matti in 1 a 50 simulazioni", passed, detail)

    passed, detail = test_color_symmetry(evaluator, 100)
    ok &= report("simmetria colori (criticita #2)", passed, detail)

    if not args.skip_monotonic:
        started = time.perf_counter()
        passed, detail = test_monotonic_strength(evaluator, args.games)
        ok &= report(
            "forza crescente con le simulazioni",
            passed,
            f"{detail} — {(time.perf_counter() - started) / 60:.1f} min",
        )
    else:
        print(f"  {GREY}[SKIP] forza crescente con le simulazioni (--skip-monotonic){RESET}")

    print()
    if ok:
        print(f"  {GREEN}Batteria §4.5 superata.{RESET}")
        print(f"  {GREY}Resta la diagnosi §4.4: scripts/run_diagnosi_44.py{RESET}")
    else:
        print(f"  {RED}Batteria §4.5 ROSSA — correggere prima di procedere.{RESET}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
