"""Gate 2 — valutazione della baseline.

Il test che conta e la simmetria (criticita #2): la stessa proprieta tornera nel backup
MCTS al Gate 5, dove sara molto piu difficile da diagnosticare. Verificarla qui, su un
codice di cento righe, e il momento giusto.
"""

from __future__ import annotations

import chess
import pytest

from chessbot.baseline.evaluation import (
    KING_TABLE_ENDGAME,
    KING_TABLE_MIDGAME,
    MATE_SCORE,
    PIECE_SQUARE_TABLES,
    PIECE_VALUES,
    evaluate,
)

from .conftest_positions import random_positions

pytestmark = pytest.mark.fast


def test_valori_materiali_come_da_piano():
    """§Fase 1: P=1, N=B=3, R=5, Q=9 — in centipedoni."""
    assert PIECE_VALUES[chess.PAWN] == 100
    assert PIECE_VALUES[chess.KNIGHT] == 320
    assert PIECE_VALUES[chess.BISHOP] == 330
    assert PIECE_VALUES[chess.ROOK] == 500
    assert PIECE_VALUES[chess.QUEEN] == 900


def test_tabelle_hanno_64_caselle():
    for piece_type, table in PIECE_SQUARE_TABLES.items():
        assert len(table) == 64, f"tabella di {piece_type} ha {len(table)} caselle"
    assert len(KING_TABLE_MIDGAME) == 64
    assert len(KING_TABLE_ENDGAME) == 64


# --------------------------------------------------------------------------------------
# Criticita #2 — simmetria
# --------------------------------------------------------------------------------------


def test_posizione_iniziale_vale_zero():
    """Simmetrica per costruzione: qualunque valore diverso da 0 e un bug nelle PST."""
    assert evaluate(chess.Board()) == 0


def test_invarianza_per_specchiatura_completa():
    """`board.mirror()` specchia posizione, colori E tratto.

    E quindi la STESSA situazione vista dall'altro lato: in convenzione negamax la
    valutazione deve essere UGUALE, non opposta. Confondere questa con l'antisimmetria
    del solo tratto (test sotto) e un errore facile da fare leggendo il piano di fretta.
    """
    checked = 0
    for board in random_positions(n_games=30, seed=808):
        assert evaluate(board) == evaluate(board.mirror()), f"asimmetria su {board.fen()}"
        checked += 1
    assert checked >= 1_000


def test_antisimmetria_per_cambio_di_tratto():
    """Stessa posizione, tratto invertito: la valutazione deve cambiare segno.

    E la forma in cui la criticita #2 si presentera nel backup MCTS.

    Le posizioni terminali sono escluse, e non e un modo di aggirare il test: matto e
    stallo hanno un punteggio ASSOLUTO (rispettivamente -MATE_SCORE e 0) che non
    dipende dalla valutazione posizionale. Invertire il tratto puo trasformare una
    posizione normale in uno stallo, e in quel caso i due punteggi non sono
    confrontabili per costruzione.
    """
    checked = 0
    for board in random_positions(n_games=25, seed=909):
        flipped = board.copy()
        flipped.turn = not flipped.turn
        if not flipped.is_valid():
            continue
        if board.is_game_over() or flipped.is_game_over():
            continue
        assert evaluate(board) == -evaluate(flipped), f"asimmetria su {board.fen()}"
        checked += 1
    assert checked >= 1_000


# --------------------------------------------------------------------------------------
# Verifica incrociata con il conteggio materiale a mano
# --------------------------------------------------------------------------------------


def test_vantaggio_materiale_riconosciuto():
    """Una donna in piu deve valere piu di una donna, al netto delle PST."""
    board = chess.Board("4k3/8/8/8/8/8/8/3QK3 w - - 0 1")
    assert evaluate(board) > 800

    board = chess.Board("3qk3/8/8/8/8/8/8/4K3 w - - 0 1")
    assert evaluate(board) < -800


def test_confronto_con_conteggio_materiale_a_mano():
    """Verifica incrociata: l'eval segue il conteggio materiale fatto a mano.

    Si parte da una posizione con materiale gia pari e non banale, e si aggiunge un
    pezzo: la differenza deve valere esattamente valore + bonus posizionale. Cosi il
    contributo di tutto il resto si cancella nella sottrazione.

    Solo pezzi leggeri: aggiungere torre o donna porterebbe il vantaggio alla soglia
    che attiva `_mate_drive`, e la differenza non sarebbe piu il solo materiale. Che il
    termine di matto entri esattamente li e voluto, ed e verificato da
    `test_mate_drive_spinge_il_re_debole_al_bordo`.
    """
    base = chess.Board("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1")
    for piece_type in (chess.KNIGHT, chess.BISHOP):
        with_piece = base.copy()
        with_piece.set_piece_at(chess.D4, chess.Piece(piece_type, chess.WHITE))
        delta = evaluate(with_piece) - evaluate(base)
        expected = PIECE_VALUES[piece_type] + PIECE_SQUARE_TABLES[piece_type][chess.D4]
        assert delta == expected, f"materiale non lineare per {piece_type}"


def test_matto_e_stallo():
    matto = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    assert matto.is_checkmate()
    assert evaluate(matto) == -MATE_SCORE

    stallo = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    assert stallo.is_stalemate()
    assert evaluate(stallo) == 0


def test_materiale_insufficiente_e_patta():
    board = chess.Board("4k3/8/8/8/8/8/8/4KB2 w - - 0 1")
    assert board.is_insufficient_material()
    assert evaluate(board) == 0


def test_re_in_centro_premiato_nel_finale():
    """Nel finale il re deve preferire il centro all'angolo.

    Serve un pedone sulla scacchiera: re contro re e materiale insufficiente, e vale 0
    per regolamento a prescindere da dove stanno i re.
    """
    angolo = chess.Board("7k/8/8/8/8/8/P7/K7 w - - 0 1")
    centro = chess.Board("7k/8/8/3K4/8/8/P7/8 w - - 0 1")
    assert evaluate(centro) > evaluate(angolo)


def test_mate_drive_spinge_il_re_debole_al_bordo():
    """Con un vantaggio decisivo, il re avversario al bordo deve valere di piu.

    Senza questo incentivo la baseline vince materiale e poi non conclude: e il bug
    per cui KR vs K finiva in patta per la regola delle 50 mosse.
    """
    centro = chess.Board("8/8/8/4k3/8/8/8/3QK3 w - - 0 1")
    bordo = chess.Board("7k/8/8/8/8/8/8/3QK3 w - - 0 1")
    assert evaluate(bordo) > evaluate(centro)
