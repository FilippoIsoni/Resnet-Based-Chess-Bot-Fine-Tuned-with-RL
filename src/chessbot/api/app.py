"""FastAPI app — Stadio 7 backend (§docs/API_CONTRACT.md).

Due endpoint: `GET /health` (sempre reattivo, anche a ricerca in corso) e
`POST /move` (CPU-bound, serializzato con un flag su `EngineState`, eseguito
in un thread separato per non bloccare l'event loop — vedi `EngineState.busy`
in `engine.py`).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

import chess
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from .deps import get_engine_state
from .engine import LEVEL_SEARCH_CONFIGS, EngineState, load_model, new_start_time
from .inference import build_move_response
from .rate_limit import InMemoryRateLimiter
from .schemas import HealthResponse, MoveRequest, MoveResponse

# Origine di GitHub Pages autorizzata esplicitamente; qualsiasi porta locale
# per lo sviluppo. Mai "*" insieme a credenziali (qui non servono comunque) —
# trappola documentata in docs/API_CONTRACT.md.
ALLOWED_ORIGINS = ["https://filippoisoni.github.io"]
LOCALHOST_ORIGIN_REGEX = r"^http://localhost:\d+$"

EngineDep = Annotated[EngineState, Depends(get_engine_state)]


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Carica il modello una volta, in modo sincrono e bloccante, prima di
    `yield`. Costa 1-2s (48MB su CPU): trascurabile rispetto ai ~30s di cold
    start dell'intero container HF Spaces, quindi non serve caricarlo in
    background con un lock aggiuntivo per proteggere lo swap."""
    evaluator, device = load_model()
    app.state.engine = EngineState(
        evaluator=evaluator,
        device=device,
        model_loaded=evaluator is not None,
        start_time=new_start_time(),
        rate_limiter=InMemoryRateLimiter(),
    )
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="chessbot API", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_origin_regex=LOCALHOST_ORIGIN_REGEX,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=HealthResponse)
    async def health(engine: EngineDep) -> HealthResponse:
        return HealthResponse(
            model_loaded=engine.model_loaded,
            device=str(engine.device),
            uptime_s=time.monotonic() - engine.start_time,
        )

    @app.post("/move", response_model=MoveResponse)
    async def move(req: MoveRequest, request: Request, engine: EngineDep) -> MoveResponse:
        if not engine.rate_limiter.check(_client_ip(request)):
            raise HTTPException(status_code=429, detail="troppe richieste, riprova tra poco")

        if engine.evaluator is None:
            raise HTTPException(
                status_code=503,
                detail="modello non disponibile",
                headers={"Retry-After": "5"},
            )
        evaluator = engine.evaluator  # variabile locale: narrowing esplicito per mypy

        # Nessun `await` fra questo controllo e l'assegnazione sotto: nel
        # modello a singolo event loop di asyncio e atomico, quindi basta un
        # flag booleano per serializzare le ricerche senza una coda vera.
        if engine.busy:
            raise HTTPException(
                status_code=503,
                detail="ricerca gia in corso",
                headers={"Retry-After": "2"},
            )
        engine.busy = True
        try:
            board = chess.Board(req.fen)  # gia validato dal field_validator di MoveRequest
            config = LEVEL_SEARCH_CONFIGS[req.level]
            return await asyncio.to_thread(build_move_response, board, evaluator, config)
        finally:
            engine.busy = False

    return app


app = create_app()

__all__ = ["app", "create_app"]
