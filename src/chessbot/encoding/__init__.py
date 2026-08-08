"""Encoding posizione e mosse — il fondamento del progetto (Stadio 1).

Tutto il resto ci si appoggia: la rete vede solo quello che questo package produce.
Un bug qui non crasha, degrada silenziosamente la forza del motore — per questo il
Gate 1 e il piu severo della pipeline (§2.5 del piano).

Uso tipico nel preprocessing:

    from chessbot.encoding import encode_sample
    tensore, indice = encode_sample(board, mossa_giocata)

`encode_sample` orienta insieme scacchiera ed etichetta, che e il modo di non incorrere
nella criticita #1.
"""

from .board import (
    BOARD_SIZE,
    N_PLANES,
    PIECE_ORDER,
    decode_move,
    decode_planes_to_fen,
    encode_board,
    encode_move,
    encode_sample,
    legal_move_mask,
)
from .digest import tensor_digest
from .moves import (
    N_MOVE_TYPES,
    N_MOVES,
    MoveEncodingError,
    index_to_move,
    mirror_move_index,
    move_to_index,
)

__all__ = [
    "BOARD_SIZE",
    "N_MOVES",
    "N_MOVE_TYPES",
    "N_PLANES",
    "PIECE_ORDER",
    "MoveEncodingError",
    "decode_move",
    "decode_planes_to_fen",
    "encode_board",
    "encode_move",
    "encode_sample",
    "index_to_move",
    "legal_move_mask",
    "mirror_move_index",
    "move_to_index",
    "tensor_digest",
]
