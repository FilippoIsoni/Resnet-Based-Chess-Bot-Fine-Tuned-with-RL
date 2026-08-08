"""Generatore di posizioni per i test dell'encoding.

Non e un `conftest.py`: e un modulo importato esplicitamente, cosi il campione e
riproducibile e ogni test dichiara il seed che usa. Un test che dipende da posizioni
casuali non seedate fallisce a intermittenza, ed e peggio di un test assente.
"""

from __future__ import annotations

import random
from collections.abc import Iterator

import chess

# Posizioni scelte a mano che il gioco casuale non raggiunge quasi mai: promozioni,
# en passant, arrocchi ancora disponibili, finali stretti.
EDGE_CASE_FENS: tuple[str, ...] = (
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "8/P6k/8/8/8/8/7K/8 w - - 0 1",
    "8/7K/8/8/8/8/p6k/8 b - - 0 1",
    "1n5k/P7/8/8/8/8/7K/8 w - - 0 1",
    "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
    "rnbqkbnr/pppp1ppp/8/8/3pP3/5N2/PPP2PPP/RNBQKB1R b KQkq e3 0 3",
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
    "4k3/8/8/8/8/8/8/4K2R w K - 0 1",
)


def random_positions(
    *, n_games: int = 40, seed: int = 1337, max_fullmoves: int = 80
) -> Iterator[chess.Board]:
    """Genera posizioni giocando partite casuali, piu i casi limite fissi.

    Le partite casuali danno varieta; i casi limite coprono cio che il caso non produce.
    """
    rng = random.Random(seed)

    for fen in EDGE_CASE_FENS:
        yield chess.Board(fen)

    for _ in range(n_games):
        board = chess.Board()
        while not board.is_game_over() and board.fullmove_number <= max_fullmoves:
            yield board.copy(stack=False)
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
