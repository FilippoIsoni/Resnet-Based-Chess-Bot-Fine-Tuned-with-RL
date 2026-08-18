"""Rate limiting minimale per `/move` (~30 richieste/minuto per IP, §API_CONTRACT.md).

In-memory, per singola istanza: lo Space gira a singola replica (un solo
worker Uvicorn, stessa assunzione di `EngineState.busy` in `engine.py`), quindi
non serve Redis o uno store condiviso fra processi.

Applicato solo a `/move`, non a `/health`: quest'ultimo va chiamato liberamente
per il risveglio/polling durante il cold start di HF Spaces senza consumare
quota (vedi `docs/API_CONTRACT.md`).
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class InMemoryRateLimiter:
    """Finestra scorrevole per chiave (tipicamente l'IP del client)."""

    max_requests: int = 30
    window_s: float = 60.0
    _hits: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))

    def check(self, key: str) -> bool:
        """True se la richiesta e ammessa (e la registra); False se va rifiutata."""
        now = time.monotonic()
        hits = self._hits[key]
        cutoff = now - self.window_s
        while hits and hits[0] < cutoff:
            hits.pop(0)
        if len(hits) >= self.max_requests:
            return False
        hits.append(now)
        return True


__all__ = ["InMemoryRateLimiter"]
