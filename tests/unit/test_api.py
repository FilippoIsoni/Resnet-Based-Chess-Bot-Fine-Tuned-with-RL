"""Gate 7-backend — API FastAPI (§docs/API_CONTRACT.md, PIPELINE.md).

Tutti i test usano un evaluator finto iniettato via
`app.dependency_overrides[get_engine_state]`: nessun checkpoint reale viene
mai caricato — requisito esplicito del gate ("i test fast non caricano 48 MB
di pesi").
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterator

import chess
import httpx
import pytest
import torch
from fastapi.testclient import TestClient

from chessbot.api import engine as engine_module
from chessbot.api.app import create_app
from chessbot.api.deps import get_engine_state
from chessbot.api.engine import EngineState
from chessbot.api.inference import build_move_response
from chessbot.api.rate_limit import InMemoryRateLimiter
from chessbot.search import SearchConfig

pytestmark = pytest.mark.fast

FOOLS_MATE_FEN = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
STALEMATE_FEN = "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"


class ScriptedEvaluator:
    """Priori uniformi, valore costante e noto — per testare il segno di eval
    e per simulare un costo CPU-bound (`delay_s`) senza una rete vera."""

    def __init__(self, value: float = 0.0, delay_s: float = 0.0) -> None:
        self.value = value
        self.delay_s = delay_s

    def evaluate(self, boards: list[chess.Board]) -> list[tuple[dict[chess.Move, float], float]]:
        if self.delay_s:
            time.sleep(self.delay_s)
        out = []
        for board in boards:
            moves = list(board.legal_moves)
            uniform = 1.0 / max(len(moves), 1)
            out.append(({m: uniform for m in moves}, self.value))
        return out


def make_state(**overrides: object) -> EngineState:
    defaults: dict[str, object] = {
        "evaluator": ScriptedEvaluator(0.4),
        "device": torch.device("cpu"),
        "model_loaded": True,
        "start_time": time.monotonic(),
        "rate_limiter": InMemoryRateLimiter(max_requests=1000, window_s=60.0),
    }
    defaults.update(overrides)
    return EngineState(**defaults)  # type: ignore[arg-type]


def make_client(state: EngineState) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_engine_state] = lambda: state
    return TestClient(app)


@pytest.fixture
def fake_state() -> EngineState:
    return make_state()


@pytest.fixture
def client(fake_state: EngineState) -> Iterator[TestClient]:
    with make_client(fake_state) as c:
        yield c


# --------------------------------------------------------------------------------------
# /health
# --------------------------------------------------------------------------------------


def test_health_ok(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["device"] == "cpu"
    assert body["uptime_s"] >= 0.0


def test_health_model_loaded_false_non_e_un_errore() -> None:
    """`model_loaded: false` e legittimo durante il caricamento dei pesi
    all'avvio: la UI lo interpreta come "si sta svegliando", non un errore."""
    state = make_state(evaluator=None, model_loaded=False)
    with make_client(state) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["model_loaded"] is False


# --------------------------------------------------------------------------------------
# /move — casi normali
# --------------------------------------------------------------------------------------


def test_move_posizione_normale(client: TestClient) -> None:
    resp = client.post("/move", json={"fen": chess.STARTING_FEN})
    assert resp.status_code == 200
    body = resp.json()
    board = chess.Board(chess.STARTING_FEN)
    assert chess.Move.from_uci(body["move"]) in board.legal_moves
    assert body["san"]
    assert body["game_over"] is False
    assert body["result"] is None


def test_level_default_e_medium(client: TestClient) -> None:
    resp = client.post("/move", json={"fen": chess.STARTING_FEN})
    assert resp.json()["sims"] == 200


def test_pv_non_supera_dieci_mosse(client: TestClient) -> None:
    resp = client.post("/move", json={"fen": chess.STARTING_FEN, "level": "medium"})
    assert len(resp.json()["pv"]) <= 10


# --------------------------------------------------------------------------------------
# /move — posizioni terminali (move: null senza eccezioni)
# --------------------------------------------------------------------------------------


def test_move_scacco_matto_move_null(client: TestClient) -> None:
    resp = client.post("/move", json={"fen": FOOLS_MATE_FEN})
    assert resp.status_code == 200
    body = resp.json()
    assert body["move"] is None
    assert body["san"] is None
    assert body["eval"] == pytest.approx(-1.0)
    assert body["game_over"] is True
    assert body["result"] == "0-1"
    assert body["pv"] == []
    assert body["sims"] == 0


def test_move_stallo(client: TestClient) -> None:
    resp = client.post("/move", json={"fen": STALEMATE_FEN})
    body = resp.json()
    assert body["move"] is None
    assert body["eval"] == pytest.approx(0.0)
    assert body["game_over"] is True
    assert body["result"] == "1/2-1/2"


# --------------------------------------------------------------------------------------
# Segno di eval — il criterio piu delicato del gate
# --------------------------------------------------------------------------------------


def test_segno_eval_con_evaluator_a_valore_noto() -> None:
    """Test a livello di funzione, non via HTTP: dato un evaluator a valore
    costante e noto, la risposta ha il segno atteso.

    `simulations=1` e deliberato. Con una sola simulazione la selezione
    scende esattamente di un livello dalla radice, valuta la foglia con
    l'evaluator (valore costante 0.8), e il backup nega risalendo: la radice
    riceve `-0.8`. Con piu simulazioni la ricerca scende a profondita diverse
    e — negando ad ogni livello — un evaluator a valore COSTANTE smette di
    dare un segno deterministico alla radice: non e un bug (la criticita #2
    e gia coperta da `tests/unit/test_search.py`), e solo che questo test
    specifico richiede una sola discesa per essere inequivocabile.
    """
    evaluator = ScriptedEvaluator(0.8)
    response = build_move_response(chess.Board(), evaluator, SearchConfig(simulations=1))
    assert response.eval == pytest.approx(-0.8)


# --------------------------------------------------------------------------------------
# Validazione — sempre 422, mai 500
# --------------------------------------------------------------------------------------


def test_fen_malformato_422(client: TestClient) -> None:
    resp = client.post("/move", json={"fen": "non e una fen"})
    assert resp.status_code == 422


def test_fen_illegale_422(client: TestClient) -> None:
    """Sintatticamente parsabile ma senza re: `board.is_valid()` lo respinge."""
    resp = client.post("/move", json={"fen": "8/8/8/8/8/8/8/8 w - - 0 1"})
    assert resp.status_code == 422


def test_level_fuori_dominio_422(client: TestClient) -> None:
    resp = client.post("/move", json={"fen": chess.STARTING_FEN, "level": "impossible"})
    assert resp.status_code == 422


# --------------------------------------------------------------------------------------
# 503 — motore occupato o modello assente, mai 500
# --------------------------------------------------------------------------------------


def test_motore_occupato_503() -> None:
    state = make_state(busy=True)
    with make_client(state) as client:
        resp = client.post("/move", json={"fen": chess.STARTING_FEN})
    assert resp.status_code == 503
    assert resp.headers.get("retry-after") == "2"


def test_modello_assente_503() -> None:
    state = make_state(evaluator=None, model_loaded=False)
    with make_client(state) as client:
        resp = client.post("/move", json={"fen": chess.STARTING_FEN})
    assert resp.status_code == 503
    assert resp.headers.get("retry-after") == "5"


# --------------------------------------------------------------------------------------
# Rate limiting
# --------------------------------------------------------------------------------------


def test_rate_limit_429() -> None:
    state = make_state(rate_limiter=InMemoryRateLimiter(max_requests=3, window_s=60.0))
    with make_client(state) as client:
        for _ in range(3):
            resp = client.post("/move", json={"fen": chess.STARTING_FEN})
            assert resp.status_code == 200
        resp = client.post("/move", json={"fen": chess.STARTING_FEN})
    assert resp.status_code == 429
    assert "detail" in resp.json()


def test_rate_limit_non_si_applica_a_health() -> None:
    """/health va chiamato liberamente durante il risveglio/polling di HF
    Spaces: non deve consumare la quota di /move."""
    state = make_state(rate_limiter=InMemoryRateLimiter(max_requests=1, window_s=60.0))
    with make_client(state) as client:
        for _ in range(5):
            resp = client.get("/health")
            assert resp.status_code == 200


# --------------------------------------------------------------------------------------
# CORS
# --------------------------------------------------------------------------------------


def test_cors_origine_autorizzata(client: TestClient) -> None:
    resp = client.get("/health", headers={"Origin": "https://filippoisoni.github.io"})
    assert resp.headers.get("access-control-allow-origin") == "https://filippoisoni.github.io"


def test_cors_origine_non_autorizzata(client: TestClient) -> None:
    resp = client.get("/health", headers={"Origin": "https://evil.example.com"})
    assert "access-control-allow-origin" not in resp.headers


def test_cors_localhost_qualsiasi_porta(client: TestClient) -> None:
    resp = client.get("/health", headers={"Origin": "http://localhost:54321"})
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:54321"


# --------------------------------------------------------------------------------------
# Concorrenza — il criterio piu delicato del Gate 7-backend
# --------------------------------------------------------------------------------------


def test_health_risponde_durante_ricerca_lunga(
    fake_state: EngineState, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`/health` deve rispondere MENTRE una ricerca MCTS e in corso — se
    l'handler di `/move` bloccasse l'event loop, questo test lo scoprirebbe.

    Usa `httpx.AsyncClient` su `ASGITransport` per lanciare `/move` e
    `/health` sullo STESSO event loop con `asyncio.create_task`. Due
    `TestClient` sincroni separati girerebbero su event loop distinti e
    passerebbero anche se `/move` fosse scritta come `def` sincrona
    bloccante — il bug esatto che questo test deve intercettare.
    """
    monkeypatch.setitem(engine_module.LEVEL_SEARCH_CONFIGS, "hard", SearchConfig(simulations=80))
    fake_state.evaluator = ScriptedEvaluator(0.0, delay_s=0.03)

    async def scenario() -> tuple[int, int, float]:
        app = create_app()
        app.dependency_overrides[get_engine_state] = lambda: fake_state
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as ac:
            move_task = asyncio.create_task(
                ac.post("/move", json={"fen": chess.STARTING_FEN, "level": "hard"})
            )
            await asyncio.sleep(0.05)  # lascia partire la ricerca
            t0 = time.perf_counter()
            health_resp = await ac.get("/health")
            elapsed = time.perf_counter() - t0
            move_resp = await move_task
        return health_resp.status_code, move_resp.status_code, elapsed

    health_status, move_status, health_elapsed = asyncio.run(scenario())
    assert health_status == 200
    assert move_status == 200
    assert health_elapsed < 0.5, (
        f"/health ha impiegato {health_elapsed:.2f}s mentre /move era in corso — "
        "l'event loop e probabilmente bloccato"
    )
