"""Verifica che le mosse del dataset siano SENSATE, non solo legali (Gate 3).

    python scripts/verify_move_quality.py --samples 300
    python scripts/verify_move_quality.py --samples 1000 --depth 12 --split val

Sostituisce con una misura il criterio del Gate 3 che chiedeva un giudizio a occhio:

> ispezione manuale di 20 campioni... nessun gate automatico sostituisce questo controllo

Il giudizio "questa mossa ha senso" richiede di saper giocare a scacchi. Qui lo fa
Stockfish, su un campione molto piu grande di venti posizioni.

## Perche serve, dato che `verify_alignment.py` da gia 100%

Quello verifica la **correttezza**: la mossa e legale nella sua posizione. Ma un
preprocessing che pescasse sistematicamente la mossa sbagliata-ma-legale — poniamo, la
seconda mossa legale invece di quella giocata — passerebbe al 100% e produrrebbe
etichette di puro rumore. La rete convergerebbe comunque, su niente.

## Il metodo: confronto controllato

Un numero assoluto non direbbe nulla ("il 40% delle mosse coincide con Stockfish" e
tanto o poco?). Si misurano quindi DUE cose sulle STESSE posizioni:

- il **dataset**: le mosse davvero giocate, da partite con Elo medio >= 2000
- il **baseline casuale**: una mossa legale a caso

Se le etichette sono buone, il dataset deve staccare nettamente il caso su tutte le
metriche. Se i due valori si somigliano, le etichette sono rumore — ed e esattamente il
bug che l'ispezione manuale doveva intercettare.

Le metriche:

| Metrica | Significato |
|---|---|
| top-1 | la mossa coincide con la prima scelta di Stockfish |
| top-3 | e fra le prime tre |
| perdita media | quanti centipedoni si perdono rispetto alla mossa migliore |
| errori gravi | mosse che perdono piu di 300 centipedoni |
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import chess  # noqa: E402
import chess.engine  # noqa: E402

from chessbot.data import SPLITS, iter_shard_boards, read_shard  # noqa: E402
from chessbot.encoding import decode_move  # noqa: E402

_COLOR = sys.stdout.isatty()
GREEN, RED, YELLOW, GREY, BOLD, RESET = (
    ("\033[32m", "\033[31m", "\033[33m", "\033[90m", "\033[1m", "\033[0m")
    if _COLOR
    else ("", "", "", "", "", "")
)

STOCKFISH = ROOT / "tools" / "stockfish" / "stockfish.exe"

# Oltre questa perdita in centipedoni la mossa e un errore grave, non un'imprecisione.
BLUNDER_CP = 300

# Le posizioni con pochissime mosse legali sono banali (spesso una sola risposta allo
# scacco) e gonfierebbero il top-1 di entrambi i contendenti, mascherando la differenza.
MIN_LEGAL_MOVES = 5


@dataclass
class Scores:
    """Metriche accumulate per un giocatore (dataset o casuale)."""

    name: str
    top1: int = 0
    top3: int = 0
    blunders: int = 0
    losses: list[int] = field(default_factory=list)
    total: int = 0

    def add(self, rank: int | None, loss_cp: int) -> None:
        self.total += 1
        if rank == 0:
            self.top1 += 1
        if rank is not None and rank < 3:
            self.top3 += 1
        if loss_cp >= BLUNDER_CP:
            self.blunders += 1
        self.losses.append(loss_cp)

    @property
    def top1_pct(self) -> float:
        return 100 * self.top1 / self.total if self.total else 0.0

    @property
    def top3_pct(self) -> float:
        return 100 * self.top3 / self.total if self.total else 0.0

    @property
    def blunder_pct(self) -> float:
        return 100 * self.blunders / self.total if self.total else 0.0

    @property
    def mean_loss(self) -> float:
        return statistics.mean(self.losses) if self.losses else 0.0

    @property
    def median_loss(self) -> float:
        return statistics.median(self.losses) if self.losses else 0.0


def sample_positions(
    data_dir: Path, split: str, n: int, rng: random.Random
) -> list[tuple[chess.Board, chess.Move]]:
    """Estrae n coppie (posizione, mossa) a caso da uno split."""
    shards = sorted((data_dir / split).glob("shard_*.npy"))
    if not shards:
        return []

    out: list[tuple[chess.Board, chess.Move]] = []
    # Si pesca da shard diversi per non concentrarsi su poche partite consecutive.
    while len(out) < n:
        path = rng.choice(shards)
        array = read_shard(path)
        for _ in range(min(64, n - len(out))):
            i = rng.randrange(len(array))
            board, move_index, _wdl, _eval = next(iter_shard_boards(array[i : i + 1]))
            if board.legal_moves.count() < MIN_LEGAL_MOVES:
                continue
            try:
                move = decode_move(board, move_index)
            except Exception:
                continue
            if move in board.legal_moves:
                out.append((board, move))
    return out[:n]


def rank_and_loss(infos: list, board: chess.Board, move: chess.Move) -> tuple[int | None, int]:
    """Posizione della mossa nella classifica di Stockfish, e perdita in centipedoni.

    `infos` e l'output di `analyse(..., multipv=N)`, gia ordinato dal migliore.
    """
    best_score = infos[0]["score"].pov(board.turn).score(mate_score=10_000)

    rank = None
    move_score = None
    for i, info in enumerate(infos):
        pv = info.get("pv")
        if pv and pv[0] == move:
            rank = i
            move_score = info["score"].pov(board.turn).score(mate_score=10_000)
            break

    if move_score is None:
        # Fuori dalle prime N: si valuta la posizione dopo la mossa, che e il costo vero.
        return None, -1  # segnalato al chiamante, che fara l'analisi dedicata
    return rank, max(0, best_score - move_score)


def evaluate_move(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    move: chess.Move,
    infos: list,
    limit: chess.engine.Limit,
) -> tuple[int | None, int]:
    """Rank e perdita, analizzando a parte le mosse fuori dalla multipv."""
    rank, loss = rank_and_loss(infos, board, move)
    if loss >= 0:
        return rank, loss

    best_score = infos[0]["score"].pov(board.turn).score(mate_score=10_000)
    after = board.copy()
    after.push(move)
    info = engine.analyse(after, limit)
    # Dopo la mossa tocca all'avversario: il punteggio va negato per tornare al nostro pov.
    move_score = -info["score"].pov(after.turn).score(mate_score=10_000)
    return None, max(0, best_score - move_score)


def verdict(dataset: Scores, baseline: Scores) -> tuple[bool, list[str]]:
    """Il dataset deve staccare nettamente il caso su tutte le metriche."""
    lines: list[str] = []

    top1_ratio = dataset.top1_pct / baseline.top1_pct if baseline.top1_pct else float("inf")
    loss_ratio = baseline.mean_loss / dataset.mean_loss if dataset.mean_loss else float("inf")

    ok = (
        dataset.top1_pct > 3 * max(baseline.top1_pct, 1.0)
        and dataset.mean_loss < baseline.mean_loss / 2
        and dataset.blunder_pct < baseline.blunder_pct / 2
    )

    if ok:
        lines.append(f"{GREEN}Le mosse del dataset sono sensate.{RESET}")
        lines.append(
            f"Coincidono con la prima scelta di Stockfish {top1_ratio:.1f} volte piu"
            f" spesso del caso, e perdono {loss_ratio:.1f} volte meno materiale."
        )
        lines.append(
            "Sono mosse di giocatori forti, non etichette rumorose: il preprocessing"
            " associa a ogni posizione la mossa davvero giocata."
        )
    else:
        lines.append(f"{RED}Le mosse del dataset non si distinguono abbastanza dal caso.{RESET}")
        lines.append(
            "Sospetto che il preprocessing associ alle posizioni una mossa sbagliata"
            " (ma legale). Ispezionare `samples_from_game` prima di allenare."
        )
    return ok, lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "processed")
    ap.add_argument("--split", choices=SPLITS, default="val")
    ap.add_argument("--samples", type=int, default=300)
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--multipv", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--engine", type=Path, default=STOCKFISH)
    args = ap.parse_args()

    if not args.engine.exists():
        print(f"Stockfish non trovato: {args.engine}")
        print("Scaricarlo da https://stockfishchess.org/download/ in tools/stockfish/")
        return 1

    rng = random.Random(args.seed)
    print(f"{BOLD}Qualita delle mosse del dataset{RESET}")
    print(f"  split {args.split}, {args.samples} posizioni, Stockfish depth {args.depth}")
    print()

    positions = sample_positions(args.data, args.split, args.samples, rng)
    if not positions:
        print(f"nessuna posizione in {args.data / args.split}")
        return 1

    engine = chess.engine.SimpleEngine.popen_uci(str(args.engine.resolve()))
    limit = chess.engine.Limit(depth=args.depth)

    dataset = Scores("dataset")
    baseline = Scores("casuale")

    started = time.perf_counter()
    last_report = 0.0
    try:
        for i, (board, move) in enumerate(positions):
            infos = engine.analyse(board, limit, multipv=args.multipv)

            dataset.add(*evaluate_move(engine, board, move, infos, limit))

            # Stessa posizione, mossa a caso: il confronto deve essere appaiato.
            random_move = rng.choice(list(board.legal_moves))
            baseline.add(*evaluate_move(engine, board, random_move, infos, limit))

            now = time.perf_counter()
            if now - last_report >= 20:
                print(
                    f"  {i + 1}/{len(positions)} — dataset top-1 {dataset.top1_pct:.1f}%,"
                    f" casuale {baseline.top1_pct:.1f}%",
                    flush=True,
                )
                last_report = now
    finally:
        engine.quit()

    elapsed = time.perf_counter() - started

    print()
    print(f"{BOLD}Confronto su {dataset.total} posizioni{RESET}")
    print(f"  {'metrica':22s} {'dataset':>12s} {'casuale':>12s}")
    print("  " + "-" * 48)
    print(f"  {'top-1 Stockfish':22s} {dataset.top1_pct:>11.1f}% {baseline.top1_pct:>11.1f}%")
    print(f"  {'top-3 Stockfish':22s} {dataset.top3_pct:>11.1f}% {baseline.top3_pct:>11.1f}%")
    print(f"  {'perdita media (cp)':22s} {dataset.mean_loss:>12.0f} {baseline.mean_loss:>12.0f}")
    print(
        f"  {'perdita mediana (cp)':22s} {dataset.median_loss:>12.0f} {baseline.median_loss:>12.0f}"
    )
    print(
        f"  {'errori gravi (>300cp)':22s} {dataset.blunder_pct:>11.1f}%"
        f" {baseline.blunder_pct:>11.1f}%"
    )
    print(f"  {GREY}{elapsed / 60:.1f} min{RESET}")

    print()
    print(f"{BOLD}Verdetto{RESET}")
    ok, lines = verdict(dataset, baseline)
    for line in lines:
        print(f"  {line}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
