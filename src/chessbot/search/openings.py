"""Aperture randomizzate per il self-play (§5.3, criticita #7).

> Man mano che la rete migliora, le partite fra due copie identiche diventano in
> maggioranza patte e il segnale evapora.

Due copie della stessa rete, dalla stessa posizione iniziale, giocano quasi la stessa
partita. Il rumore di Dirichlet e la temperatura aiutano, ma non bastano: la varieta va
messa **all'inizio**, facendo partire ogni partita da una posizione diversa.

E cio che fa Leela, ed e il rimedio che §5.3 chiede di applicare **insieme** agli altri.

## Da dove vengono le posizioni

Non da un file Polyglot esterno, ma **dal dataset che gia abbiamo**: 20 milioni di
posizioni giocate da umani sopra i 2000 Elo. Pescare le posizioni dopo 8 semimosse da li
significa partire da aperture reali, nelle proporzioni in cui si giocano davvero.

Vantaggi rispetto a un `.bin` scaricato:

- nessuna dipendenza esterna da procurarsi e versionare
- le aperture sono quelle del dominio su cui la rete e stata allenata
- la distribuzione e naturale: le linee popolari compaiono piu spesso, come nella realta

Il libro si genera una volta e si salva in JSON: e piccolo (poche centinaia di FEN) e
leggibile, quindi si puo ispezionare a occhio.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import chess
import numpy as np

# Semimosse dopo le quali si considera "finita l'apertura". Il piano dice 4-8 (§5.3);
# 8 da posizioni gia caratterizzate ma ancora equilibrate.
DEFAULT_PLIES = 8

# Sotto questa frequenza nel dataset una posizione e una stranezza, non un'apertura.
MIN_OCCURRENCES = 3


def build_book_from_dataset(
    data_dir: Path,
    *,
    plies: int = DEFAULT_PLIES,
    max_positions: int = 500,
    shards_to_scan: int = 4,
    min_occurrences: int = MIN_OCCURRENCES,
) -> list[str]:
    """Estrae le posizioni di apertura piu frequenti dal dataset.

    Scansiona alcuni shard e tiene le posizioni raggiunte esattamente dopo `plies`
    semimosse. Il conteggio delle occorrenze serve a scartare le linee rare: un'apertura
    vista tre volte su centomila posizioni non e teoria, e un errore di qualcuno.

    Returns:
        Lista di FEN, ordinate per frequenza decrescente.
    """
    from chessbot.data import iter_shard_boards, read_shard

    counter: Counter[str] = Counter()

    # > **Non si puo usare `board.ply()`.** Lo storage non conserva il numero di mossa
    # > (§2.6: sarebbero due byte per posizione su venti milioni), quindi ogni board
    # > ricostruita riparte da `fullmove_number=1` e `ply()` vale sempre 0 o 1.
    # >
    # > Si riconosce l'apertura dai **pezzi ancora sulla scacchiera**: dopo 8 semimosse
    # > se ne sono catturati pochissimi, e i pedoni sono quasi tutti al loro posto. E un
    # > criterio approssimato ma robusto, e per scegliere posizioni di partenza basta.
    for split in ("train",):
        paths = sorted((data_dir / split).glob("shard_*.npy"))[:shards_to_scan]
        for path in paths:
            array = read_shard(path)
            for board, _move_index, _wdl, _eval in iter_shard_boards(array):
                if not _looks_like_opening(board, plies):
                    continue
                counter[board.fen()] += 1

    return [fen for fen, count in counter.most_common(max_positions) if count >= min_occurrences]


def _looks_like_opening(board: chess.Board, plies: int) -> bool:
    """Euristica: la posizione e plausibilmente a fine apertura?

    Dopo ~8 semimosse una partita normale ha ancora tutti o quasi tutti i pezzi, e i
    pedoni sono ancora vicini alle case di partenza. Si escludono anche le posizioni
    iniziali pure, che non aggiungono varieta.
    """
    pieces = len(board.piece_map())
    if pieces < 30:  # gia catturati piu di due pezzi: siamo oltre l'apertura
        return False
    if pieces == 32 and board.fullmove_number == 1 and board.castling_rights == chess.BB_CORNERS:
        # Posizione iniziale o quasi: nessuna varieta da offrire.
        moved = sum(
            1
            for sq, p in board.piece_map().items()
            if p.piece_type == chess.PAWN and chess.square_rank(sq) not in (1, 6)
        )
        if moved < 2:
            return False

    # I pedoni fuori dalle traverse 2 e 7 sono quelli che si sono mossi: dopo `plies`
    # semimosse ce ne sono tipicamente 2-5.
    advanced = sum(
        1
        for sq, p in board.piece_map().items()
        if p.piece_type == chess.PAWN and chess.square_rank(sq) not in (1, 6)
    )
    return 2 <= advanced <= plies


def save_book(path: Path, fens: list[str], *, source: str = "") -> None:
    """Salva il libro in JSON, leggibile e ispezionabile."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "_comment": (
            "Posizioni di apertura per il self-play (§5.3, criticita #7). "
            "Estratte dal dataset: sono aperture giocate davvero da umani 2000+ Elo."
        ),
        "source": source,
        "plies": DEFAULT_PLIES,
        "positions": len(fens),
        "fens": fens,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")


def load_book(path: Path) -> list[chess.Board]:
    """Carica il libro come lista di board pronte all'uso."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [chess.Board(fen) for fen in data["fens"]]


def sample_openings(book: list[chess.Board], n: int, rng: np.random.Generator) -> list[chess.Board]:
    """Pesca `n` posizioni di partenza dal libro, con ripetizione.

    Con ripetizione di proposito: il libro ha centinaia di posizioni e le iterazioni ne
    chiedono migliaia. Ripescare la stessa apertura va bene — le partite divergono
    comunque per il rumore di Dirichlet e la temperatura.
    """
    if not book:
        return [chess.Board() for _ in range(n)]
    idx = rng.integers(len(book), size=n)
    return [book[int(i)].copy() for i in idx]


def book_diversity(book: list[chess.Board]) -> dict[str, int]:
    """Quante aperture distinte ci sono, per famiglia di prima mossa.

    Serve a verificare a occhio che il libro non sia sbilanciato: se il 90% delle
    posizioni viene da 1.e4, la varieta e minore di quanto il conteggio suggerisca.
    """
    families: Counter[str] = Counter()
    for board in book:
        # Ricostruisce la prima mossa dalla sequenza: la board del libro non ha storico,
        # quindi si guarda quale pedone o pezzo si e mosso per primo.
        first = "altro"
        if board.piece_at(chess.E4) == chess.Piece(chess.PAWN, chess.WHITE):
            first = "1.e4"
        elif board.piece_at(chess.D4) == chess.Piece(chess.PAWN, chess.WHITE):
            first = "1.d4"
        elif board.piece_at(chess.C4) == chess.Piece(chess.PAWN, chess.WHITE):
            first = "1.c4"
        elif board.piece_at(chess.F3) == chess.Piece(chess.KNIGHT, chess.WHITE):
            first = "1.Nf3"
        families[first] += 1
    return dict(families.most_common())


__all__ = [
    "DEFAULT_PLIES",
    "book_diversity",
    "build_book_from_dataset",
    "load_book",
    "sample_openings",
    "save_book",
]
