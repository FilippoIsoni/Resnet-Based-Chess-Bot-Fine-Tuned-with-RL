"""Storage compatto delle posizioni (§2.6 del piano).

**Non si salvano i tensori espansi.** 20M x 19 x 64 x 4 byte = 97 GB. Si salva la forma
compatta e si espande al volo nel DataLoader:

| Campo | Tipo | Byte |
|---|---|---|
| 12 bitboard dei pezzi | `uint64` x 12 | 96 |
| stato (arrocco, en passant, tratto) | `uint16` | 2 |
| halfmove clock + ripetizioni | `uint16` | 2 |
| indice mossa (0-4671) | `uint16` | 2 |
| esito WDL | `uint8` | 1 |
| eval Stockfish in centipedoni | `int16` | 2 |
| **totale** | | **105** |

~2 GB per 20M posizioni, contro 97 GB. Gli shard sono `.npy` con un dtype strutturato:
si aprono con `np.load(..., mmap_mode="r")` e il sistema operativo pagina cio che serve,
senza caricare lo shard intero.

> Il round-trip deve essere **bit a bit** (criterio del Gate 3): si scrive uno shard, lo
> si rilegge e i campi devono coincidere esattamente. Un errore di packing qui non
> crasha, corrompe silenziosamente una frazione del dataset.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

import chess
import numpy as np

from .pgn import Sample

# Ordine dei bitboard: 6 tipi di pezzo x 2 colori. Fissato — cambiarlo invalida gli
# shard gia scritti, che non hanno modo di accorgersene.
PIECE_TYPES = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)

# Valore sentinella per "nessuna valutazione Stockfish disponibile". Sta fuori dal range
# plausibile di un eval reale, che il parser limita a +/-10.000 centipedoni.
NO_EVAL = np.int16(-32768)

# Bit del campo `state`.
_CASTLE_WK, _CASTLE_WQ, _CASTLE_BK, _CASTLE_BQ = 1, 2, 4, 8
_TURN_BLACK = 16
_EP_SHIFT = 5  # 4 bit: colonna 0-7, oppure 8 = nessun en passant
_EP_NONE = 8

SAMPLE_DTYPE = np.dtype(
    [
        ("bitboards", np.uint64, (12,)),
        ("state", np.uint16),
        ("clock", np.uint16),
        ("move_index", np.uint16),
        ("wdl", np.uint8),
        ("eval_cp", np.int16),
    ]
)


def pack_board(board: chess.Board) -> tuple[np.ndarray, int, int]:
    """Comprime una posizione in (bitboard, state, clock)."""
    bitboards = np.zeros(12, dtype=np.uint64)
    for i, piece_type in enumerate(PIECE_TYPES):
        bitboards[i] = np.uint64(int(board.pieces(piece_type, chess.WHITE)))
        bitboards[6 + i] = np.uint64(int(board.pieces(piece_type, chess.BLACK)))

    state = 0
    if board.has_kingside_castling_rights(chess.WHITE):
        state |= _CASTLE_WK
    if board.has_queenside_castling_rights(chess.WHITE):
        state |= _CASTLE_WQ
    if board.has_kingside_castling_rights(chess.BLACK):
        state |= _CASTLE_BK
    if board.has_queenside_castling_rights(chess.BLACK):
        state |= _CASTLE_BQ
    if board.turn == chess.BLACK:
        state |= _TURN_BLACK

    ep = _EP_NONE if board.ep_square is None else chess.square_file(board.ep_square)
    state |= ep << _EP_SHIFT

    # Il clock oltre 100 non aggiunge informazione: la partita e patta a 100.
    clock = min(board.halfmove_clock, 255)
    return bitboards, state, clock


def unpack_board(bitboards: np.ndarray, state: int, clock: int) -> chess.Board:
    """Ricostruisce la posizione. Inverso esatto di `pack_board`."""
    board = chess.Board.empty()
    for i, piece_type in enumerate(PIECE_TYPES):
        for color, offset in ((chess.WHITE, 0), (chess.BLACK, 6)):
            bb = int(bitboards[offset + i])
            for square in chess.SquareSet(bb):
                board.set_piece_at(square, chess.Piece(piece_type, color))

    castling = ""
    if state & _CASTLE_WK:
        castling += "K"
    if state & _CASTLE_WQ:
        castling += "Q"
    if state & _CASTLE_BK:
        castling += "k"
    if state & _CASTLE_BQ:
        castling += "q"
    board.set_castling_fen(castling or "-")

    board.turn = chess.BLACK if state & _TURN_BLACK else chess.WHITE

    ep_file = (state >> _EP_SHIFT) & 0b1111
    if ep_file != _EP_NONE:
        # La casella di presa al varco sta sulla traversa 6 se muove il Bianco, 3 se il Nero.
        rank = 5 if board.turn == chess.WHITE else 2
        board.ep_square = chess.square(ep_file, rank)

    board.halfmove_clock = int(clock)
    return board


def encode_samples(samples: Iterable[Sample]) -> np.ndarray:
    """Comprime una sequenza di campioni in un array strutturato.

    L'indice mossa e quello ORIENTATO (`encoding.encode_move`), quindi gia coerente con
    il tensore che il DataLoader produrra: la criticita #1 e risolta qui una volta sola,
    invece che ad ogni epoca.
    """
    from chessbot.encoding import encode_move

    rows: list[tuple] = []
    for sample in samples:
        board = chess.Board(sample.fen)
        bitboards, state, clock = pack_board(board)
        move = chess.Move.from_uci(sample.move_uci)
        rows.append(
            (
                bitboards,
                state,
                clock,
                encode_move(board, move),
                sample.wdl,
                NO_EVAL
                if sample.eval_cp is None
                else np.int16(np.clip(sample.eval_cp, -32767, 32767)),
            )
        )
    return np.array(rows, dtype=SAMPLE_DTYPE)


def write_shard(path: Path, samples: Iterable[Sample]) -> int:
    """Scrive uno shard `.npy`. Restituisce il numero di posizioni scritte."""
    array = encode_samples(samples)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array, allow_pickle=False)
    return len(array)


def read_shard(path: Path, *, mmap: bool = True) -> np.ndarray:
    """Rilegge uno shard. Con `mmap` non lo carica in memoria."""
    return np.load(path, mmap_mode="r" if mmap else None, allow_pickle=False)


def iter_shard_boards(array: np.ndarray) -> Iterator[tuple[chess.Board, int, int, int | None]]:
    """Scorre uno shard restituendo (board, indice mossa, wdl, eval)."""
    for row in array:
        board = unpack_board(row["bitboards"], int(row["state"]), int(row["clock"]))
        eval_cp = None if row["eval_cp"] == NO_EVAL else int(row["eval_cp"])
        yield board, int(row["move_index"]), int(row["wdl"]), eval_cp


def bytes_per_sample() -> int:
    """Dimensione effettiva di un campione su disco, per i conti di §2.6."""
    return SAMPLE_DTYPE.itemsize
