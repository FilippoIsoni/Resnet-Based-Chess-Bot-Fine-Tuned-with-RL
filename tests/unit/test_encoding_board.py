"""Gate 1 — encoder della posizione (§2.3, §2.5).

La batteria obbligatoria del piano, piu i controlli sul confronto con i file golden.
Ogni test qui verifica l'encoder contro una fonte INDIPENDENTE dall'encoder stesso:
`python-chess` come oracolo, oppure il golden versionato.
"""

from __future__ import annotations

import json
from pathlib import Path

import chess
import numpy as np
import pytest

from chessbot.encoding import (
    BOARD_SIZE,
    N_MOVES,
    N_PLANES,
    decode_move,
    decode_planes_to_fen,
    encode_board,
    encode_move,
    encode_sample,
    legal_move_mask,
    tensor_digest,
)

from .conftest_positions import random_positions

pytestmark = pytest.mark.fast

GOLDEN_PATH = Path(__file__).resolve().parents[1] / "golden" / "positions.json"


def _fen_head(fen: str) -> str:
    """Primi cinque campi della FEN: quelli che il tensore contiene davvero.

    Il numero di mossa non e codificato di proposito (§2.3), quindi confrontarlo
    sarebbe un falso fallimento.
    """
    return " ".join(fen.split()[:5])


def _expected_fen_head(board: chess.Board) -> str:
    """FEN attesa dalla ricostruzione, con l'en passant nella forma grezza.

    `Board.fen()` cancella il campo en passant quando nessun pedone puo davvero
    catturare al varco; `fen(en_passant="fen")` lo conserva. L'encoder codifica la
    colonna cosi come la dichiara la posizione, quindi il confronto va fatto sulla
    stessa convenzione da entrambi i lati — altrimenti si confrontano due
    normalizzazioni diverse e il test fallisce senza che ci sia un bug.
    """
    return _fen_head(board.fen(en_passant="fen"))


# --------------------------------------------------------------------------------------
# Forma e invarianti di base
# --------------------------------------------------------------------------------------


def test_forma_del_tensore():
    planes = encode_board(chess.Board())
    assert planes.shape == (N_PLANES, BOARD_SIZE, BOARD_SIZE)
    assert planes.dtype == np.float32


def test_somma_piani_pezzi_uguale_numero_pezzi():
    """Criterio del Gate 1: nessun pezzo perso o duplicato dall'encoder."""
    for board in random_positions(n_games=40, seed=11):
        planes = encode_board(board, repetitions=0)
        assert planes[0:12].sum() == pytest.approx(len(board.piece_map()))


def test_ogni_casella_ha_al_piu_un_pezzo():
    """Due piani accesi sulla stessa casella significherebbe un pezzo duplicato."""
    for board in random_positions(n_games=20, seed=12):
        planes = encode_board(board, repetitions=0)
        assert planes[0:12].sum(axis=0).max() <= 1.0


def test_nessun_piano_per_il_turno():
    """Orientando dal lato che muove, il tensore non deve distinguere il colore.

    Una posizione e la sua speculare-a-colori-invertiti sono la STESSA cosa per la rete.
    """
    board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    assert np.array_equal(encode_board(board), encode_board(board.mirror()))


def test_valori_solo_in_zero_uno_per_i_piani_binari():
    for board in random_positions(n_games=15, seed=13):
        planes = encode_board(board, repetitions=0)
        binari = planes[0:17]
        assert np.isin(binari, (0.0, 1.0)).all()


# --------------------------------------------------------------------------------------
# Verifica incrociata: ricostruzione della FEN (§2.5)
# --------------------------------------------------------------------------------------


def test_ricostruzione_fen_dal_tensore():
    """Encoder vs re-parsing: nessuna deriva su 1000+ posizioni (criterio del Gate 1).

    E il check piu forte della batteria: se dal tensore si ricostruisce la stessa FEN,
    l'encoder non ha perso ne inventato informazione.
    """
    checked = 0
    for board in random_positions(n_games=40, seed=1234):
        planes = encode_board(board, repetitions=0)
        rebuilt = decode_planes_to_fen(planes, side_to_move=board.turn)
        assert _fen_head(rebuilt) == _expected_fen_head(board), f"deriva su {board.fen()}"
        checked += 1
    assert checked >= 1_000, f"campione troppo piccolo: {checked}"


# --------------------------------------------------------------------------------------
# Criticita #1 — simmetria specchiata
# --------------------------------------------------------------------------------------


def test_simmetria_specchiata_tensore_e_indice():
    """Il test che il piano chiama "criticita #1".

    Una posizione con il Bianco al tratto e la stessa posizione specchiata con il Nero
    al tratto devono produrre tensori IDENTICI e indici mossa CORRETTAMENTE specchiati.
    Passare la prima meta e fallire la seconda e esattamente il bug silenzioso.
    """
    checked = 0
    for board in random_positions(n_games=30, seed=4242):
        mirrored = board.mirror()
        assert np.array_equal(
            encode_board(board, repetitions=0), encode_board(mirrored, repetitions=0)
        ), f"tensori diversi per {board.fen()}"

        for move in board.legal_moves:
            mirrored_move = chess.Move(
                chess.square_mirror(move.from_square),
                chess.square_mirror(move.to_square),
                promotion=move.promotion,
            )
            assert mirrored_move in mirrored.legal_moves
            assert encode_move(board, move) == encode_move(mirrored, mirrored_move)
            checked += 1
    assert checked >= 5_000


def test_encode_sample_orienta_insieme_input_ed_etichetta():
    board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1")
    board.push_uci("e7e5")
    board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

    white_tensor, white_index = encode_sample(board, chess.Move.from_uci("e2e4"))
    black_board = board.mirror()
    black_tensor, black_index = encode_sample(black_board, chess.Move.from_uci("e7e5"))

    assert np.array_equal(white_tensor, black_tensor)
    assert white_index == black_index


def test_decode_move_inverte_encode_move():
    checked = 0
    for board in random_positions(n_games=30, seed=99):
        for move in board.legal_moves:
            assert decode_move(board, encode_move(board, move)) == move
            checked += 1
    assert checked >= 10_000


# --------------------------------------------------------------------------------------
# Piani non posizionali
# --------------------------------------------------------------------------------------


def test_piani_arrocco():
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    planes = encode_board(board)
    assert planes[12].all() and planes[13].all() and planes[14].all() and planes[15].all()

    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w - - 0 1")
    planes = encode_board(board)
    assert not planes[12].any() and not planes[15].any()


def test_arrocco_prospettiva_di_chi_muove():
    """I piani 12-13 sono sempre "i miei" diritti, anche quando muove il Nero."""
    board = chess.Board("r3k2r/8/8/8/8/8/8/4K3 b kq - 0 1")
    planes = encode_board(board)
    assert planes[12].all(), "il corto del Nero deve stare nel piano 'mio corto'"
    assert planes[13].all()
    assert not planes[14].any() and not planes[15].any()


def test_piano_en_passant():
    board = chess.Board("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3")
    planes = encode_board(board)
    acceso = np.nonzero(planes[16].any(axis=0))[0]
    assert len(acceso) == 1 and acceso[0] == chess.square_file(chess.F6)

    board = chess.Board()
    assert not encode_board(board)[16].any()


def test_piano_halfmove_clock():
    board = chess.Board("8/8/4k3/8/8/4K3/8/8 w - - 50 120")
    assert encode_board(board)[17] == pytest.approx(0.5)

    board = chess.Board("8/8/4k3/8/8/4K3/8/8 w - - 0 1")
    assert encode_board(board)[17].sum() == 0.0


def test_piano_halfmove_clock_troncato_a_uno():
    board = chess.Board("8/8/4k3/8/8/4K3/8/8 w - - 150 200")
    assert encode_board(board)[17] == pytest.approx(1.0)


def test_piano_ripetizioni():
    board = chess.Board()
    assert encode_board(board, repetitions=0)[18].sum() == 0.0
    assert encode_board(board, repetitions=3)[18] == pytest.approx(1.0)


# --------------------------------------------------------------------------------------
# Maschera delle mosse legali
# --------------------------------------------------------------------------------------


def test_maschera_legale_conta_e_posiziona_giusto():
    for board in random_positions(n_games=20, seed=31):
        mask = legal_move_mask(board)
        assert mask.shape == (N_MOVES,)
        assert mask.sum() == board.legal_moves.count()
        for move in board.legal_moves:
            assert mask[encode_move(board, move)]


# --------------------------------------------------------------------------------------
# File golden (§2.5: 20 posizioni note con tensore atteso)
# --------------------------------------------------------------------------------------


def test_golden_presente_e_completo():
    assert GOLDEN_PATH.exists(), "eseguire: python scripts/gen_golden.py"
    data = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    assert len(data["positions"]) >= 20
    assert data["n_planes"] == N_PLANES and data["n_moves"] == N_MOVES


def test_golden_corrisponde_all_encoder_attuale():
    """Confronto con i 20 tensori di riferimento versionati.

    Questo test deve dare lo stesso esito su Windows e macOS: e la verifica incrociata
    fra le due macchine prevista dallo Stadio 0-bis.
    """
    data = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    for entry in data["positions"]:
        board = chess.Board(entry["fen"])
        move = chess.Move.from_uci(entry["move_uci"])
        planes = encode_board(board, repetitions=0)

        assert tensor_digest(planes) == entry["tensor_sha256"], (
            f"[{entry['name']}] il tensore e cambiato rispetto al golden"
        )
        assert encode_move(board, move) == entry["move_index"], (
            f"[{entry['name']}] l'indice mossa e cambiato rispetto al golden"
        )
        assert _fen_head(
            decode_planes_to_fen(planes, side_to_move=board.turn)
        ) == _expected_fen_head(board)


def test_golden_promozioni_donna_e_sottopromozione_su_indici_diversi():
    """Lettura mirata del golden: la criticita #4 vista dai dati di riferimento."""
    data = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    by_name = {e["name"]: e for e in data["positions"]}
    donna = by_name["promozione_donna_bianco"]["move_index"]
    cavallo = by_name["sottopromozione_cavallo_bianco"]["move_index"]
    assert donna != cavallo
    assert donna % 73 < 56 <= 64 <= cavallo % 73
