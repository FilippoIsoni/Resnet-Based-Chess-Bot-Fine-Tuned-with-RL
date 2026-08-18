"""API FastAPI per il deployment (Stadio 7 — backend, §docs/API_CONTRACT.md).

    from chessbot.api import create_app
    app = create_app()

Riusa invariati `Evaluator`/`run_mcts` del motore (torch CPU, niente ONNX —
vedi docs/DECISIONS.md). Non gioca da solo: implementa solo `GET /health` e
`POST /move` secondo il contratto scritto con la UI Flutter.
"""

from .app import create_app

__all__ = ["create_app"]
