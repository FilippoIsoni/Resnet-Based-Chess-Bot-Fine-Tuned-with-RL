"""Gate 2 — ricerca minimax e perft.

I test lenti (100 partite contro random, perft profondo) stanno in
`tests/integration/test_gate2_baseline.py`: qui restano solo quelli che girano ad ogni
commit sotto i 20 secondi complessivi (livello L1).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import chess
import pytest

from chessbot.baseline.evaluation import MATE_SCORE
from chessbot.baseline.match import MatchResult, play_game, random_player
from chessbot.baseline.perft import PERFT_SUITE, perft, perft_divide
from chessbot.baseline.search import best_move, search

pytestmark = pytest.mark.fast

MATE_IN_1_PATH = Path(__file__).resolve().parents[1] / "golden" / "mate_in_1.json"


# --------------------------------------------------------------------------------------
# Perft — verifica indipendente della generazione mosse
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("name,fen,counts", PERFT_SUITE, ids=[p[0] for p in PERFT_SUITE])
def test_perft_fino_a_profondita_3(name: str, fen: str, counts: tuple[int, ...]):
    """Conteggi esatti dal Chess Programming Wiki: fonte totalmente indipendente."""
    board = chess.Board(fen)
    for depth in range(1, min(4, len(counts))):
        assert perft(board, depth) == counts[depth], f"{name} depth {depth}"


def test_perft_divide_somma_al_totale():
    board = chess.Board()
    divided = perft_divide(board, 3)
    assert sum(divided.values()) == perft(board, 3)
    assert len(divided) == 20


def test_perft_zero_e_uno():
    board = chess.Board()
    assert perft(board, 0) == 1
    assert perft(board, 1) == 20


# --------------------------------------------------------------------------------------
# Matto in 1 — criterio del Gate 2
# --------------------------------------------------------------------------------------


def test_dataset_matto_in_1_valido():
    """Le etichette del dataset sono verificate da python-chess, non date per buone."""
    data = json.loads(MATE_IN_1_PATH.read_text(encoding="utf-8"))
    assert len(data) >= 50
    for entry in data:
        board = chess.Board(entry["fen"])
        assert entry["mates"], f"nessuna mossa di matto per {entry['fen']}"
        for uci in entry["mates"]:
            move = chess.Move.from_uci(uci)
            assert move in board.legal_moves
            board.push(move)
            assert board.is_checkmate(), f"{uci} non e matto in {entry['fen']}"
            board.pop()


def test_trova_matto_in_1_su_50_posizioni():
    """Criterio esplicito del Gate 2: 50/50.

    Profondita 1: per un matto in UNA mossa basta, ed e molto piu rapido. Che il
    motore lo trovi anche scendendo e verificato da `test_preferisce_il_matto_piu_vicino`.
    """
    data = json.loads(MATE_IN_1_PATH.read_text(encoding="utf-8"))[:50]
    trovati = 0
    for entry in data:
        board = chess.Board(entry["fen"])
        result = search(board, depth=1)
        if result.move is not None and result.move.uci() in entry["mates"]:
            trovati += 1
    assert trovati == 50, f"trovati solo {trovati}/50 matti in 1"


def test_punteggio_di_matto_riconoscibile():
    data = json.loads(MATE_IN_1_PATH.read_text(encoding="utf-8"))[:10]
    for entry in data:
        result = search(chess.Board(entry["fen"]), depth=1)
        assert result.is_mate, f"matto non segnalato in {entry['fen']}"
        assert result.score > MATE_SCORE - 1000


def test_preferisce_il_matto_piu_vicino():
    """Il termine `ply` deve rendere un matto in 1 migliore di uno in 2."""
    board = chess.Board("6k1/5ppp/8/8/8/8/8/R3R1K1 w - - 0 1")
    result = search(board, depth=4)
    board.push(result.move)
    assert board.is_checkmate(), "con torre e torre il matto in 1 esiste e va giocato"


# --------------------------------------------------------------------------------------
# Proprieta della ricerca
# --------------------------------------------------------------------------------------


def test_restituisce_sempre_una_mossa_legale():
    rng = random.Random(4)
    board = chess.Board()
    for _ in range(20):
        if board.is_game_over():
            break
        move = best_move(board, depth=2, rng=rng)
        assert move is not None and move in board.legal_moves
        board.push(move)


def test_nessuna_mossa_in_posizione_terminale():
    matto = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    assert search(matto, depth=2).move is None


def test_non_modifica_la_board_passata():
    board = chess.Board()
    fen_prima = board.fen()
    search(board, depth=3)
    assert board.fen() == fen_prima


def test_cattura_il_pezzo_libero():
    """Sanita elementare: una donna indifesa va presa."""
    board = chess.Board("4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1")
    move = best_move(board, depth=3)
    assert move == chess.Move.from_uci("e4d5")


def test_profondita_maggiore_non_peggiora_la_valutazione_tattica():
    """Con piu profondita il motore non deve perdere una tattica che gia vedeva."""
    board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4")
    for depth in (1, 2):
        move = best_move(board, depth=depth)
        assert move is not None and move in board.legal_moves


def test_profondita_zero_rifiutata():
    with pytest.raises(ValueError):
        search(chess.Board(), depth=0)


def test_le_statistiche_sono_popolate():
    result = search(chess.Board(), depth=3)
    assert result.stats.nodes > 0
    assert result.stats.elapsed > 0
    assert result.stats.nps > 0


def test_alpha_beta_riduce_i_nodi():
    """Sanita dell'ordinamento: i cutoff devono esserci davvero."""
    result = search(chess.Board(), depth=4)
    assert result.stats.cutoffs > 0


def test_le_mosse_pari_merito_sono_davvero_equivalenti():
    """Regressione del bug della finestra stretta alla radice.

    `search` raccoglie le mosse di pari punteggio e ne estrae una a caso quando le
    viene passato un `rng`. Perche abbia senso, i punteggi devono essere valori VERI:
    con la finestra `(-INFINITY, -alpha)` i rami potati tornavano il limite di cutoff
    invece del loro valore, e mosse perdenti finivano nel gruppo delle migliori.

    Qui si verifica su una posizione dove esiste una sola mossa buona: rilanciando la
    ricerca con seed diversi deve uscire sempre quella. Prima del fix uscivano mosse
    diverse, tutte credute equivalenti.
    """
    board = chess.Board("4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1")
    scelte = {best_move(board, depth=2, rng=random.Random(seed)) for seed in range(12)}
    assert scelte == {chess.Move.from_uci("e4d5")}, f"mosse non equivalenti scelte: {scelte}"


def test_il_punteggio_della_radice_e_un_valore_vero():
    """Il punteggio restituito deve coincidere con quello della mossa giocata.

    Verifica incrociata: si gioca la mossa scelta e si valuta la posizione risultante
    con una ricerca indipendente di un livello meno profondo. I due numeri devono
    coincidere, perche sono la stessa quantita calcolata in due modi.
    """
    for fen in (
        "4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1",
        "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    ):
        board = chess.Board(fen)
        result = search(board, depth=3)
        board.push(result.move)
        controprova = search(board, depth=2)
        assert result.score == -controprova.score, f"punteggio incoerente su {fen}"


# --------------------------------------------------------------------------------------
# Infrastruttura di match
# --------------------------------------------------------------------------------------


def test_partita_contro_random_termina():
    """Partita corta: il match completo del Gate 2 sta in tests/integration/."""
    rng = random.Random(11)
    outcome, board = play_game(lambda b: best_move(b, 2, rng=rng), random_player(rng), max_plies=40)
    assert outcome in {"1-0", "0-1", "1/2-1/2"}
    assert board.ply() > 0


def test_match_result_intervallo_di_confidenza():
    """Regola #1: il margine deve crescere quando le partite diminuiscono."""
    tante = MatchResult(wins=110, losses=90, draws=0)
    poche = MatchResult(wins=11, losses=9, draws=0)
    assert tante.games == 200 and poche.games == 20
    assert tante.elo_margin < poche.elo_margin
    assert "±" in tante.summary()


def test_match_result_risultato_estremo_non_inventa_un_elo():
    cappotto = MatchResult(wins=100, losses=0, draws=0)
    assert cappotto.score_rate == 1.0
    assert cappotto.elo_diff == float("inf")
    assert "non stimabile" in cappotto.summary()


def test_match_result_punteggio_con_patte():
    r = MatchResult(wins=10, losses=5, draws=5)
    assert r.games == 20
    assert r.score == 12.5
    assert r.score_rate == 0.625
