"""Test del classificatore di `scripts/verify_alignment.py`.

> Se il classificatore non distingue i tre casi noti — allineato, specchiato, sfasato di
> un ply — i suoi risultati sul dataset non valgono niente. Questi test costruiscono i
> tre casi a mano e pretendono la categoria giusta.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import chess
import pytest

from chessbot.encoding import encode_move

pytestmark = pytest.mark.fast

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "verify_alignment.py"


def _load_module():
    """Carica lo script come modulo: e un eseguibile, non un package importabile."""
    spec = importlib.util.spec_from_file_location("verify_alignment", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["verify_alignment"] = module
    spec.loader.exec_module(module)
    return module


va = _load_module()


def test_caso_allineato_da_ok():
    """Coppia corretta: la mossa e legale nella sua posizione."""
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    category, decoded = va.classify(board, encode_move(board, move), None, None)
    assert category == "ok"
    assert decoded == move


def test_caso_allineato_col_nero_al_tratto():
    """Il caso che il mirroring rende non banale: con il nero al tratto l'indice e
    orientato, e `decode_move` deve riapplicare il flip."""
    board = chess.Board()
    board.push_uci("e2e4")
    move = chess.Move.from_uci("e7e5")
    category, decoded = va.classify(board, encode_move(board, move), None, None)
    assert category == "ok"
    assert decoded == move


def test_caso_specchiato_da_ok_mirror():
    """Indice NON orientato dove servirebbe: la mossa e legale solo specchiata.

    E il bug che si presenterebbe se il flip venisse applicato zero volte in scrittura
    e una in lettura (o viceversa).
    """
    from chessbot.encoding import move_to_index

    board = chess.Board()
    board.push_uci("g1f3")  # tocca al nero

    # Una mossa del nero il cui indice NON orientato decodifica in qualcosa di
    # illegale-ma-legale-specchiato.
    trovato = None
    for candidate in board.legal_moves:
        raw = move_to_index(candidate)  # indice grezzo, senza flip: e il bug
        category, _ = va.classify(board, raw, None, None)
        if category == "ok_mirror":
            trovato = candidate
            break

    assert trovato is not None, "nessun caso ok_mirror costruibile: il test non discrimina"


def _mossa_solo_in(target: chess.Board, altra: chess.Board) -> chess.Move:
    """Una mossa legale in `target` e NON in `altra`.

    Serve a costruire un caso che discrimini davvero: molte mosse (lo sviluppo dei
    cavalli, le spinte di pedone) sono legali in entrambe le posizioni e non
    proverebbero nulla.
    """
    for move in target.legal_moves:
        if move not in altra.legal_moves:
            return move
    raise AssertionError("nessuna mossa discriminante: le due posizioni sono troppo simili")


def test_caso_sfasato_da_ok_next_ply():
    """Mossa presa dal ply successivo: deve essere classificata `ok_next_ply`.

    Le due posizioni devono avere lo STESSO tratto, altrimenti il flip dell'indice
    rimappa la mossa su una legale dell'altro colore e il caso diventa `ok` — cosa
    corretta, ma che non prova nulla sullo sfasamento.
    """
    corrente = chess.Board()
    corrente.push_uci("e2e4")
    corrente.push_uci("e7e5")  # tocca al bianco

    successiva = corrente.copy()
    successiva.push_uci("g1f3")
    successiva.push_uci("b8c6")  # due ply dopo: tocca di nuovo al bianco

    mossa_futura = _mossa_solo_in(successiva, corrente)
    category, _ = va.classify(corrente, encode_move(successiva, mossa_futura), None, successiva)
    assert category == "ok_next_ply", f"{mossa_futura} classificata {category}"


def test_caso_sfasato_da_ok_prev_ply():
    precedente = chess.Board()
    precedente.push_uci("e2e4")  # tocca al nero

    corrente = precedente.copy()
    corrente.push_uci("e7e5")
    corrente.push_uci("g1f3")  # tocca di nuovo al nero

    mossa_passata = _mossa_solo_in(precedente, corrente)
    category, _ = va.classify(corrente, encode_move(precedente, mossa_passata), precedente, None)
    assert category == "ok_prev_ply", f"{mossa_passata} classificata {category}"


def test_caso_illegale_da_illegal():
    """Una mossa che non e legale ne cosi, ne specchiata, ne nei ply adiacenti."""
    board = chess.Board("8/8/8/4k3/8/8/8/4K3 w - - 0 1")
    # Indice di una mossa di donna che in questa posizione non ha alcun pezzo da muovere.
    from chessbot.encoding import move_to_index

    index = move_to_index(chess.Move.from_uci("a1a8"))
    category, _ = va.classify(board, index, None, None)
    assert category == "illegal"


def test_le_categorie_sono_mutuamente_esclusive():
    """`ok` ha la precedenza: una mossa legale non deve finire in un'altra categoria."""
    board = chess.Board()
    board.push_uci("e2e4")
    move = chess.Move.from_uci("e7e5")
    successiva = board.copy()
    successiva.push_uci("e7e5")

    category, _ = va.classify(board, encode_move(board, move), None, successiva)
    assert category == "ok"


# --------------------------------------------------------------------------------------
# Controllo del tratto
# --------------------------------------------------------------------------------------


def test_turn_matches_piece():
    board = chess.Board()
    assert va.turn_matches_piece(board, chess.Move.from_uci("e2e4"))
    # Muovere un pezzo nero quando tocca al bianco: incoerente.
    assert not va.turn_matches_piece(board, chess.Move.from_uci("e7e5"))


# --------------------------------------------------------------------------------------
# Diagnosi
# --------------------------------------------------------------------------------------


def test_diagnosi_dataset_allineato():
    from collections import Counter

    counts = Counter({"ok": 1_000_000})
    testo = " ".join(va.diagnose(counts, 1_000_000, 0))
    assert "allineato" in testo


def test_diagnosi_mirror_dominante():
    from collections import Counter

    counts = Counter({"ok": 100, "ok_mirror": 900})
    testo = " ".join(va.diagnose(counts, 1000, 0))
    assert "flip" in testo.lower()
    assert "NON i dati" in testo


def test_diagnosi_off_by_one():
    from collections import Counter

    counts = Counter({"ok": 400, "ok_next_ply": 600})
    testo = " ".join(va.diagnose(counts, 1000, 0))
    assert "off-by-one" in testo.lower()
    assert "rigenerare" in testo.lower()


def test_diagnosi_segnala_il_tratto_errato():
    from collections import Counter

    counts = Counter({"ok": 1000})
    testo = " ".join(va.diagnose(counts, 1000, 7))
    assert "colore sbagliato" in testo


def test_soglia_di_uscita():
    assert va.OK_THRESHOLD == 0.99
