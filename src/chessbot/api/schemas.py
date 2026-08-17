"""Modelli Pydantic della richiesta/risposta, secondo `docs/API_CONTRACT.md`.

La validazione del FEN vive qui, in un `field_validator`: cosi sia un FEN
malformato sia uno sintatticamente valido ma illegale (senza re, pedoni
sull'ottava traversa...) producono automaticamente un 422 nel formato
standard di FastAPI, senza bisogno di un exception handler dedicato nella
route. `chess.Board` fa da oracolo indipendente, come in tutto il resto del
progetto.
"""

from __future__ import annotations

from typing import Literal

import chess
from pydantic import BaseModel, Field, field_validator

Level = Literal["easy", "medium", "hard"]


class MoveRequest(BaseModel):
    fen: str
    level: Level = "medium"
    session: str | None = None
    """Identificatore opaco della UI. Non abilita alcun riuso dell'albero MCTS
    (nessuna decisione presa in tal senso, vedi docs/DECISIONS.md): serve solo
    per rate-limit/log futuri."""

    @field_validator("fen")
    @classmethod
    def fen_valido(cls, v: str) -> str:
        try:
            board = chess.Board(v)
        except ValueError as e:
            raise ValueError(f"FEN malformato: {e}") from e
        if not board.is_valid():
            raise ValueError(
                "FEN sintatticamente corretto ma posizione illegale (oracolo: python-chess)"
            )
        return v


class MoveResponse(BaseModel):
    move: str | None
    """Mossa scelta in UCI, o None se il FEN ricevuto era gia terminale."""

    san: str | None
    eval: float = Field(ge=-1.0, le=1.0)
    """Dal punto di vista di chi deve muovere nel FEN ricevuto (convenzione
    negamax). Nessuna conversione qui: la fa la UI."""

    pv: list[str]
    """Linea principale in SAN, fino a 10 mosse."""

    ms: int
    sims: int
    game_over: bool
    """True se la partita e finita DOPO la mossa restituita (o se il FEN
    ricevuto era gia terminale)."""

    result: str | None
    """'1-0' | '0-1' | '1/2-1/2', None se la partita continua."""


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    model_loaded: bool
    device: str
    uptime_s: float


__all__ = ["HealthResponse", "Level", "MoveRequest", "MoveResponse"]
