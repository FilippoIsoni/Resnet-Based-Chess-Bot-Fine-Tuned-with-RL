"""Gate 2 — i criteri lenti della baseline (§3, Stadio 2 di PIPELINE.md).

Girano nel gate di fase, non ad ogni commit: qui si giocano partite vere e si scende in
profondita sul perft. Il numero ufficiale del match si produce con
`scripts/run_gate2_match.py`, che lo salva in runs/ con il commit hash.
"""

from __future__ import annotations

import random

import chess
import pytest

from chessbot.baseline.match import play_match, random_player
from chessbot.baseline.perft import PERFT_SUITE, perft
from chessbot.baseline.search import best_move

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("name,fen,counts", PERFT_SUITE, ids=[p[0] for p in PERFT_SUITE])
def test_perft_profondita_4(name: str, fen: str, counts: tuple[int, ...]):
    """Perft a profondita 4: cattura bug che la profondita 3 non vede.

    Arrocco attraverso scacco ed en passant che espone il re sono i due casi classici
    che si manifestano solo scendendo.
    """
    if len(counts) <= 4:
        pytest.skip(f"{name} non ha un conteggio di riferimento a profondita 4")
    assert perft(chess.Board(fen), 4) == counts[4]


def test_batte_random_in_un_match_breve():
    """Criterio del Gate 2, versione rapida.

    Il criterio ufficiale e 100/100 e si misura con scripts/run_gate2_match.py; qui se
    ne giocano poche per intercettare una regressione senza costare minuti.
    """
    rng = random.Random(2024)
    result = play_match(
        lambda b: best_move(b, 2, rng=rng),
        random_player(rng),
        games=10,
        max_plies=160,
    )
    assert result.losses == 0, f"la baseline ha perso contro random: {result.summary()}"
    assert result.wins >= 8, f"troppo poche vittorie: {result.summary()}"


def test_profondita_maggiore_gioca_almeno_altrettanto_bene():
    """Sanita della ricerca: depth 3 non deve perdere contro depth 2.

    E la versione baseline del test di monotonicita che al Gate 5 verifichera l'MCTS
    (il test piu diagnostico dell'intero progetto). Con 10 partite il risultato ha un
    intervallo di confidenza enorme, quindi si verifica solo che non collassi.
    """
    rng = random.Random(77)
    result = play_match(
        lambda b: best_move(b, 3, rng=rng),
        lambda b: best_move(b, 2, rng=rng),
        games=10,
        max_plies=120,
    )
    assert result.wins >= result.losses, f"depth 3 peggio di depth 2: {result.summary()}"
