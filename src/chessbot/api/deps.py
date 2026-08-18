"""Punto di dependency-injection per lo stato del motore.

Esiste come simbolo stabile a cui fare
`app.dependency_overrides[get_engine_state] = ...` nei test, cosi i test non
toccano mai `app.state` direttamente ne caricano un checkpoint vero.
"""

from __future__ import annotations

from fastapi import Request

from .engine import EngineState


def get_engine_state(request: Request) -> EngineState:
    return request.app.state.engine


__all__ = ["get_engine_state"]
