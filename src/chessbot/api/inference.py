"""Il cuore logico di `POST /move` (§API_CONTRACT.md) — sincrona, CPU-bound.

Deliberatamente senza `asyncio`: e la funzione che gira dentro
`asyncio.to_thread` in `app.py`, e deve poter essere chiamata da un test
sincrono senza toccare FastAPI/TestClient (serve per il test sul segno di
`eval`, che e il punto piu delicato).

> **Caso terminale, la parte non ovvia.** Se il FEN ricevuto e gia terminale,
> `run_mcts` non fa mai un vero backup sulla radice: `Node.value` con
> `visits=0` restituisce `0.0` per definizione ("sconosciuto"), che sarebbe un
> `eval` sbagliato per una posizione di matto. Per questo si controlla
> `terminal_value(board)` PRIMA di lanciare la ricerca.
"""

from __future__ import annotations

import time

import chess

from chessbot.search import Evaluator, SearchConfig, principal_variation, run_mcts, terminal_value

from .schemas import MoveResponse


def _elapsed_ms(started: float) -> int:
    return round((time.perf_counter() - started) * 1000)


def build_move_response(
    board: chess.Board, evaluator: Evaluator, config: SearchConfig
) -> MoveResponse:
    """Cerca la mossa migliore e costruisce la risposta, secondo il contratto.

    `board` non viene modificata (si lavora su copie).
    """
    started = time.perf_counter()

    tv = terminal_value(board)
    if tv is not None:
        outcome = board.outcome(claim_draw=True)
        return MoveResponse(
            move=None,
            san=None,
            eval=tv,
            pv=[],
            ms=_elapsed_ms(started),
            sims=0,
            game_over=True,
            result=outcome.result() if outcome else None,
        )

    result = run_mcts(board, evaluator, config)
    if result.move is None:
        # Guardrail: non dovrebbe succedere (una posizione non terminale ha
        # sempre almeno una mossa legale), ma un root senza figli non deve
        # mai propagarsi come eccezione non gestita fino all'endpoint.
        outcome = board.outcome(claim_draw=True)
        return MoveResponse(
            move=None,
            san=None,
            eval=0.0,
            pv=[],
            ms=_elapsed_ms(started),
            sims=result.stats.simulations,
            game_over=True,
            result=outcome.result() if outcome else None,
        )

    san = board.san(result.move)  # va calcolato PRIMA di push()
    after = board.copy()
    after.push(result.move)

    pv_board = board.copy()
    pv_san: list[str] = []
    for move in principal_variation(result.root, board, max_depth=10):
        pv_san.append(pv_board.san(move))
        pv_board.push(move)

    # claim_draw=True: coerente con terminal_value, che tratta ripetizione e
    # regola delle 50 mosse come patte immediate (nessuna "richiesta" del
    # giocatore nel motore).
    game_over = after.is_game_over(claim_draw=True)
    outcome = after.outcome(claim_draw=True) if game_over else None

    return MoveResponse(
        move=result.move.uci(),
        san=san,
        # Radice: Q dal punto di vista di chi muove nel FEN ricevuto.
        # Nessuna conversione di segno qui — la fa la UI (docs/API_CONTRACT.md).
        eval=result.root.value,
        pv=pv_san,
        ms=_elapsed_ms(started),
        sims=result.stats.simulations,
        game_over=game_over,
        result=outcome.result() if outcome else None,
    )


__all__ = ["build_move_response"]
