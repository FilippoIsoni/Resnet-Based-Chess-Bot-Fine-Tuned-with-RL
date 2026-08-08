"""Gate 1 — codifica delle mosse (§2.4, §2.5).

Copre la biiettivita dello spazio 0-4671, la criticita #4 (promozione a donna) e la
criticita #1 (flip dell'indice), che sono i due bug silenziosi che questo stadio esiste
per intercettare.
"""

from __future__ import annotations

import chess
import pytest

from chessbot.encoding.moves import (
    N_MOVE_TYPES,
    N_MOVES,
    UNDERPROMOTION_PIECES,
    MoveEncodingError,
    index_to_move,
    mirror_move_index,
    move_to_index,
)

from .conftest_positions import random_positions

pytestmark = pytest.mark.fast


def test_spazio_indici_coerente_con_la_config():
    """4672 = 64 x 73. Se questo cambia, cambia anche configs/default.yaml."""
    assert N_MOVE_TYPES == 73
    assert N_MOVES == 4672


def test_round_trip_su_mosse_legali():
    """Round-trip move -> indice -> move: la codifica deve essere biiettiva.

    Il piano chiede >= 10.000 mosse legali con 0 fallimenti (§2.5).
    """
    total = 0
    for board in random_positions(n_games=60, seed=1337):
        for move in board.legal_moves:
            index = move_to_index(move)
            assert 0 <= index < N_MOVES, f"{move} -> {index} fuori range"
            assert index_to_move(index, board) == move, f"round-trip fallito su {move}"
            total += 1
    assert total >= 10_000, f"campione troppo piccolo: {total} mosse"


def test_indici_distinti_per_mosse_distinte():
    """Due mosse legali diverse non possono collidere sullo stesso indice."""
    for board in random_positions(n_games=25, seed=7):
        seen: dict[int, chess.Move] = {}
        for move in board.legal_moves:
            index = move_to_index(move)
            assert index not in seen, f"collisione: {move} e {seen[index]} -> {index}"
            seen[index] = move


# --------------------------------------------------------------------------------------
# Criticita #4 — promozione a donna vs sottopromozioni
# --------------------------------------------------------------------------------------


def test_promozione_a_donna_nel_blocco_mosse_da_regina():
    """La promozione a donna NON sta nei piani di sottopromozione (criticita #4)."""
    queen_move = chess.Move.from_uci("a7a8q")
    assert queen_move in chess.Board("8/P6k/8/8/8/8/7K/8 w - - 0 1").legal_moves
    move_type = move_to_index(queen_move) % N_MOVE_TYPES
    assert move_type < 56, f"promozione a donna finita nel tipo {move_type}, atteso < 56"


def test_sottopromozioni_nel_blocco_dedicato():
    board = chess.Board("8/P6k/8/8/8/8/7K/8 w - - 0 1")
    for piece, letter in zip(UNDERPROMOTION_PIECES, "nbr"):
        move = chess.Move.from_uci(f"a7a8{letter}")
        assert move in board.legal_moves
        move_type = move_to_index(move) % N_MOVE_TYPES
        assert 64 <= move_type < 73, f"sottopromozione a {letter} nel tipo {move_type}"
        assert index_to_move(move_to_index(move), board).promotion == piece


def test_le_quattro_promozioni_hanno_indici_distinti():
    """Il caso che la confusione donna/sottopromozione romperebbe."""
    for fen in ("8/P6k/8/8/8/8/7K/8 w - - 0 1", "8/7K/8/8/8/8/p6k/8 b - - 0 1"):
        board = chess.Board(fen)
        indices = {move_to_index(m) for m in board.legal_moves if m.promotion is not None}
        promotions = [m for m in board.legal_moves if m.promotion is not None]
        assert len(indices) == len(promotions) == 4


def test_promozione_con_cattura_round_trip():
    board = chess.Board("1n5k/P7/8/8/8/8/7K/8 w - - 0 1")
    for uci in ("a7b8q", "a7b8n", "a7b8b", "a7b8r"):
        move = chess.Move.from_uci(uci)
        assert move in board.legal_moves
        assert index_to_move(move_to_index(move), board) == move


# --------------------------------------------------------------------------------------
# Criticita #1 — flip dell'indice
# --------------------------------------------------------------------------------------


def test_mirror_e_involutivo_su_tutti_gli_indici():
    """mirror(mirror(i)) == i per ognuno dei 4672 indici, senza eccezioni."""
    for index in range(N_MOVES):
        assert mirror_move_index(mirror_move_index(index)) == index


def test_mirror_indice_coerente_con_mirror_mossa():
    """mirror(idx(m)) deve valere idx(m specchiata): e il cuore della criticita #1."""
    checked = 0
    for board in random_positions(n_games=30, seed=21):
        for move in board.legal_moves:
            mirrored = chess.Move(
                chess.square_mirror(move.from_square),
                chess.square_mirror(move.to_square),
                promotion=move.promotion,
            )
            assert mirror_move_index(move_to_index(move)) == move_to_index(mirrored)
            checked += 1
    assert checked >= 5_000


# --------------------------------------------------------------------------------------
# Errori
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("index", [-1, N_MOVES, N_MOVES + 1, 10**6])
def test_indice_fuori_range_solleva(index: int):
    with pytest.raises(MoveEncodingError):
        index_to_move(index)
    with pytest.raises(MoveEncodingError):
        mirror_move_index(index)


def test_mossa_nulla_solleva():
    with pytest.raises(MoveEncodingError):
        move_to_index(chess.Move(chess.E2, chess.E2))


def test_nessun_indice_fuori_range_su_tutto_il_campione():
    """Criterio esplicito del Gate 1: nessun indice generato fuori da [0, 4671]."""
    for board in random_positions(n_games=40, seed=555):
        for move in board.legal_moves:
            assert 0 <= move_to_index(move) <= 4671
