"""Stadio 5 — MCTS: albero, PUCT, backup, batching.

Test veloci su reti finte: verificano la meccanica, non la forza di gioco. Quella la
misurano `scripts/run_gate5_tests.py` e la diagnosi §4.4, che giocano partite vere.

Il test che conta di piu qui e `test_il_backup_nega_il_valore_ad_ogni_livello`: la
criticita #2 e un bug che non crasha, e questo e il punto in cui si puo intercettare in
tre righe invece che con quaranta minuti di partite.
"""

from __future__ import annotations

import json
from pathlib import Path

import chess
import numpy as np
import pytest

from chessbot.search import (
    TERMINAL_DRAW,
    TERMINAL_LOSS,
    Node,
    SearchConfig,
    add_dirichlet_noise,
    add_virtual_loss,
    backup,
    expand,
    policy_target_from,
    principal_variation,
    puct_score,
    remove_virtual_loss,
    run_mcts,
    select_child,
    select_move,
    terminal_value,
)

pytestmark = pytest.mark.fast

MATE_IN_1 = Path(__file__).resolve().parents[1] / "golden" / "mate_in_1.json"


class UniformEvaluator:
    """Rete finta: priori uniformi, valore zero. Non sa nulla di scacchi."""

    def evaluate(self, boards: list[chess.Board]) -> list[tuple[dict, float]]:
        out = []
        for board in boards:
            moves = list(board.legal_moves)
            uniform = 1.0 / max(len(moves), 1)
            out.append(({m: uniform for m in moves}, 0.0))
        return out


# --------------------------------------------------------------------------------------
# Criticita #2 — il segno del valore nel backup
# --------------------------------------------------------------------------------------


def test_il_backup_nega_il_valore_ad_ogni_livello():
    """Il test che intercetta la criticita #2 in tre righe.

    Il valore arriva dalla foglia ed e dal punto di vista di chi muove LI. Risalendo,
    il giocatore si alterna, quindi il segno deve alternarsi con lui.

    Senza la negazione il motore valuta le proprie posizioni con la prospettiva
    dell'avversario: gioca contro se stesso e sembra solo "un po' debole".
    """
    radice = Node(prior=1.0, to_play=chess.WHITE)
    figlio = Node(prior=1.0, to_play=chess.BLACK)
    nipote = Node(prior=1.0, to_play=chess.WHITE)

    backup([radice, figlio, nipote], value=1.0)

    # Il nipote e la foglia: riceve il valore cosi com'e.
    assert nipote.value_sum == pytest.approx(1.0)
    # Il figlio e l'avversario: per lui vale l'opposto.
    assert figlio.value_sum == pytest.approx(-1.0)
    # La radice torna al segno della foglia.
    assert radice.value_sum == pytest.approx(1.0)

    assert radice.visits == figlio.visits == nipote.visits == 1


def test_il_backup_accumula_su_piu_simulazioni():
    node = Node(prior=1.0, to_play=chess.WHITE)
    backup([node], 1.0)
    backup([node], -1.0)
    backup([node], 1.0)
    assert node.visits == 3
    assert node.value == pytest.approx(1.0 / 3)


# --------------------------------------------------------------------------------------
# PUCT
# --------------------------------------------------------------------------------------


def test_puct_preferisce_il_prior_alto_a_parita_di_visite():
    buono = Node(prior=0.9, to_play=chess.BLACK)
    scarso = Node(prior=0.1, to_play=chess.BLACK)
    assert puct_score(buono, 10, 2.0) > puct_score(scarso, 10, 2.0)


def test_puct_penalizza_i_rami_gia_visitati():
    """`N` sta al denominatore: piu visiti un ramo, meno vale esplorarlo ancora."""
    poco = Node(prior=0.5, to_play=chess.BLACK, visits=1, value_sum=0.0)
    molto = Node(prior=0.5, to_play=chess.BLACK, visits=50, value_sum=0.0)
    assert puct_score(poco, 100, 2.0) > puct_score(molto, 100, 2.0)


def test_puct_nega_il_valore_del_figlio():
    """Q nel figlio e dal pov dell'avversario: va negato per chi sta scegliendo.

    Un figlio che va BENE per l'avversario deve risultare poco attraente.
    """
    buono_per_lui = Node(prior=0.5, to_play=chess.BLACK, visits=10, value_sum=10.0)
    male_per_lui = Node(prior=0.5, to_play=chess.BLACK, visits=10, value_sum=-10.0)
    assert puct_score(male_per_lui, 20, 2.0) > puct_score(buono_per_lui, 20, 2.0)


def test_c_puct_regola_l_esplorazione():
    """c_puct alto favorisce l'esplorazione, basso lo sfruttamento."""
    sfruttabile = Node(prior=0.1, to_play=chess.BLACK, visits=20, value_sum=-20.0)
    inesplorato = Node(prior=0.9, to_play=chess.BLACK, visits=0)

    # Con c_puct piccolo vince il valore gia osservato.
    assert puct_score(sfruttabile, 100, 0.1) > puct_score(inesplorato, 100, 0.1)
    # Con c_puct grande vince il prior.
    assert puct_score(inesplorato, 100, 10.0) > puct_score(sfruttabile, 100, 10.0)


def test_select_child_prende_il_massimo():
    parent = Node(prior=1.0, to_play=chess.WHITE)
    parent.children = {
        chess.Move.from_uci("e2e4"): Node(prior=0.9, to_play=chess.BLACK),
        chess.Move.from_uci("d2d4"): Node(prior=0.1, to_play=chess.BLACK),
    }
    parent.visits = 10
    move, _ = select_child(parent, 2.0)
    assert move == chess.Move.from_uci("e2e4")


# --------------------------------------------------------------------------------------
# Posizioni terminali
# --------------------------------------------------------------------------------------


def test_terminale_matto():
    """Chi deve muovere e sotto matto: ha perso."""
    board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    assert board.is_checkmate()
    assert terminal_value(board) == TERMINAL_LOSS


def test_terminale_stallo():
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    assert board.is_stalemate()
    assert terminal_value(board) == TERMINAL_DRAW


def test_terminale_materiale_insufficiente():
    board = chess.Board("4k3/8/8/8/8/8/8/4KB2 w - - 0 1")
    assert terminal_value(board) == TERMINAL_DRAW


def test_posizione_normale_non_e_terminale():
    assert terminal_value(chess.Board()) is None


# --------------------------------------------------------------------------------------
# Virtual loss — il meccanismo del batching
# --------------------------------------------------------------------------------------


def test_il_virtual_loss_rende_il_ramo_meno_attraente():
    """Il ramo penalizzato deve avere un PUCT piu basso, non piu alto.

    > Regressione di un errore di segno. Il valore del nodo e dal pov di chi muove LI,
    > ma chi sceglie e il genitore, e il PUCT usa `-child.value`. Sottraendo il virtual
    > loss dal valore, la penalita diventava un PREMIO: un nodo penalizzato passava da
    > PUCT 0,1 a 1,1 e veniva riselezionato all'infinito. Misurato: 24.433 collisioni su
    > 200 simulazioni, batch medio 1,0 — il batching non funzionava affatto.
    """
    pulito = Node(prior=0.5, to_play=chess.BLACK, visits=10, value_sum=5.0)
    penalizzato = Node(prior=0.5, to_play=chess.BLACK, visits=10, value_sum=5.0)
    add_virtual_loss([penalizzato], 3)

    assert puct_score(penalizzato, 100, 2.0) < puct_score(pulito, 100, 2.0), (
        "il virtual loss deve scoraggiare il ramo, non premiarlo"
    )

    remove_virtual_loss([penalizzato], 3)
    assert penalizzato.value == pytest.approx(pulito.value)


def test_il_virtual_loss_sposta_davvero_la_selezione():
    """Il punto del virtual loss: due selezioni consecutive non devono coincidere.

    Senza questo, il batch si riempie di copie della stessa posizione e la GPU valuta N
    volte lo stesso tensore.
    """
    board = chess.Board()
    root = Node(prior=1.0, to_play=board.turn)
    priors, _ = UniformEvaluator().evaluate([board])[0]
    expand(root, board, priors)

    scelte = []
    for _ in range(4):
        move, child = select_child(root, 2.0)
        scelte.append(move)
        add_virtual_loss([root, child], 3)

    assert len(set(scelte)) == 4, f"la selezione resta bloccata su {scelte}"


def test_il_virtual_loss_si_annulla():
    node = Node(prior=1.0, to_play=chess.WHITE)
    add_virtual_loss([node], 3)
    assert node.virtual_loss == 3
    remove_virtual_loss([node], 3)
    assert node.virtual_loss == 0


# --------------------------------------------------------------------------------------
# La ricerca completa
# --------------------------------------------------------------------------------------


def test_trova_il_matto_in_1_con_rete_cieca():
    """Il criterio del Gate 5, in versione rapida.

    La rete non sa nulla: se il matto viene trovato, il merito e dell'albero.
    """
    data = json.loads(MATE_IN_1.read_text(encoding="utf-8"))[:10]
    found = 0
    for entry in data:
        board = chess.Board(entry["fen"])
        result = run_mcts(board, UniformEvaluator(), SearchConfig(simulations=50))
        if result.move is not None and result.move.uci() in entry["mates"]:
            found += 1
    assert found == len(data), f"trovati {found}/{len(data)}"


def test_il_batching_non_impedisce_di_vedere_i_terminali():
    """Regressione del bug piu grave dello Stadio 5.

    Una foglia messa in coda per la rete resta non espansa, quindi le simulazioni
    successive la riselezionavano: il batch si riempiva di duplicati invece di esplorare.

    Misurato prima del fix su un matto in 1 con 39 mosse legali e 50 simulazioni: con
    `batch_size_leaves=1` la ricerca trovava 43 posizioni terminali e il matto; con
    `batch_size_leaves=8` ne trovava ZERO e giocava a caso.
    """
    entry = json.loads(MATE_IN_1.read_text(encoding="utf-8"))[0]
    board = chess.Board(entry["fen"])

    for batch in (1, 4, 8, 24):
        result = run_mcts(
            board,
            UniformEvaluator(),
            SearchConfig(simulations=50, batch_size_leaves=batch),
        )
        assert result.move is not None
        assert result.move.uci() in entry["mates"], (
            f"con batch {batch} la ricerca non trova il matto: {result.move}"
        )
        assert result.stats.terminal_hits > 0, f"con batch {batch} nessun terminale visto"


def test_il_batching_raggruppa_davvero():
    """Il batch medio deve essere > 1, altrimenti il batching non serve a nulla."""
    result = run_mcts(
        chess.Board(),
        UniformEvaluator(),
        SearchConfig(simulations=200, batch_size_leaves=16),
    )
    assert result.stats.avg_batch > 1.5, f"batch medio {result.stats.avg_batch:.1f}"


def test_le_visite_totali_corrispondono_alle_simulazioni():
    result = run_mcts(chess.Board(), UniformEvaluator(), SearchConfig(simulations=100))
    totale = sum(c.visits for c in result.root.children.values())
    # Ogni simulazione aggiunge una visita a un figlio della radice.
    assert 90 <= totale <= 100, f"{totale} visite per 100 simulazioni"


def test_restituisce_sempre_una_mossa_legale():
    board = chess.Board()
    for _ in range(6):
        result = run_mcts(board, UniformEvaluator(), SearchConfig(simulations=30))
        assert result.move in board.legal_moves
        board.push(result.move)


def test_nessuna_mossa_in_posizione_terminale():
    matto = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    result = run_mcts(matto, UniformEvaluator(), SearchConfig(simulations=20))
    assert result.move is None


def test_non_modifica_la_board_passata():
    board = chess.Board()
    fen = board.fen()
    run_mcts(board, UniformEvaluator(), SearchConfig(simulations=30))
    assert board.fen() == fen


# --------------------------------------------------------------------------------------
# Scelta della mossa e temperatura
# --------------------------------------------------------------------------------------


def test_temperatura_zero_prende_la_piu_visitata():
    root = Node(prior=1.0, to_play=chess.WHITE)
    root.children = {
        chess.Move.from_uci("e2e4"): Node(prior=0.5, to_play=chess.BLACK, visits=100),
        chess.Move.from_uci("d2d4"): Node(prior=0.5, to_play=chess.BLACK, visits=10),
    }
    rng = np.random.default_rng(0)
    for _ in range(5):
        assert select_move(root, SearchConfig(temperature=0.0), rng) == chess.Move.from_uci("e2e4")


def test_temperatura_alta_esplora():
    """Con temperatura 1 la scelta e proporzionale alle visite, quindi varia."""
    root = Node(prior=1.0, to_play=chess.WHITE)
    root.children = {
        chess.Move.from_uci("e2e4"): Node(prior=0.5, to_play=chess.BLACK, visits=50),
        chess.Move.from_uci("d2d4"): Node(prior=0.5, to_play=chess.BLACK, visits=50),
    }
    rng = np.random.default_rng(0)
    scelte = {select_move(root, SearchConfig(temperature=1.0), rng) for _ in range(20)}
    assert len(scelte) == 2, "con temperatura 1 e visite pari dovrebbe variare"


# --------------------------------------------------------------------------------------
# Dirichlet, PV, target della policy
# --------------------------------------------------------------------------------------


def test_dirichlet_modifica_i_priori_e_ne_conserva_la_somma():
    board = chess.Board()
    root = Node(prior=1.0, to_play=chess.WHITE)
    priors, _ = UniformEvaluator().evaluate([board])[0]
    expand(root, board, priors)

    prima = {m: c.prior for m, c in root.children.items()}
    add_dirichlet_noise(root, 0.3, 0.25, np.random.default_rng(0))
    dopo = {m: c.prior for m, c in root.children.items()}

    assert prima != dopo
    assert sum(dopo.values()) == pytest.approx(1.0, abs=1e-6)


def test_dirichlet_con_epsilon_zero_non_cambia_nulla():
    """Giocando sul serio il rumore va disattivato: non deve fare nulla."""
    board = chess.Board()
    root = Node(prior=1.0, to_play=chess.WHITE)
    priors, _ = UniformEvaluator().evaluate([board])[0]
    expand(root, board, priors)

    prima = {m: c.prior for m, c in root.children.items()}
    add_dirichlet_noise(root, 0.3, 0.0, np.random.default_rng(0))
    assert {m: c.prior for m, c in root.children.items()} == prima


def test_policy_target_somma_a_uno():
    """E il target della policy nell'Expert Iteration: dev'essere una distribuzione."""
    board = chess.Board()
    result = run_mcts(board, UniformEvaluator(), SearchConfig(simulations=100))
    target = policy_target_from(board, result)
    assert target.sum() == pytest.approx(1.0, abs=1e-5)
    assert (target >= 0).all()


def test_principal_variation_e_una_sequenza_legale():
    board = chess.Board()
    result = run_mcts(board, UniformEvaluator(), SearchConfig(simulations=200))
    line = principal_variation(result.root, board, max_depth=4)

    work = board.copy()
    for move in line:
        assert move in work.legal_moves, f"{move} illegale in {work.fen()}"
        work.push(move)
