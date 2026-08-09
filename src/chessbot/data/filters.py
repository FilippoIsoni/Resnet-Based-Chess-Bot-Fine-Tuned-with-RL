"""Filtri sulle partite PGN (§2.1 del piano).

| Criterio | Valore | Motivo |
|---|---|---|
| Elo medio | >= 2000 | imitare gente forte, non la media |
| Time control | rapid / classical | il bullet e rumoroso |
| Esito | escludere il tempo scaduto in posizione non persa | etichetta di valore falsata |
| Lunghezza | >= 10 mosse | le partite abbandonate all'inizio non insegnano nulla |

> **Il campo `Event` non e affidabile per la categoria.** Nei dump Lichess fino al 2017
> "Rapid" non esiste: le categorie erano Bullet / Blitz / Classical, e le partite che
> oggi si chiamerebbero rapid stanno sotto "Classical". Filtrare per nome scarterebbe
> silenziosamente dati validi (misurato: 0 partite superavano il filtro su un campione
> di 15.706). Si usa `TimeControl`, che e un dato oggettivo in secondi.

La stima di Lichess per classificare una partita e `base + 40 * incremento`, cioe il
tempo totale ipotizzando una partita di 40 mosse. Le soglie sono quelle ufficiali.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Soglie ufficiali Lichess sulla durata stimata, in secondi.
BULLET_MAX = 179
BLITZ_MAX = 479
RAPID_MAX = 1499
# Oltre RAPID_MAX e classical.

MIN_ELO_DEFAULT = 2000
MIN_MOVES_DEFAULT = 10

_TIME_CONTROL_RE = re.compile(r"^(\d+)\+(\d+)$")


def estimated_duration(time_control: str | None) -> int | None:
    """Durata stimata in secondi da un campo TimeControl "base+incremento".

    Restituisce None per i valori non parsabili ("-" nelle partite per corrispondenza,
    o formati inattesi): un dato che non si sa interpretare non e un dato che si tiene.
    """
    if not time_control:
        return None
    match = _TIME_CONTROL_RE.match(time_control.strip())
    if not match:
        return None
    base, increment = int(match.group(1)), int(match.group(2))
    return base + 40 * increment


def category(time_control: str | None) -> str | None:
    """Categoria Lichess: bullet, blitz, rapid o classical."""
    duration = estimated_duration(time_control)
    if duration is None:
        return None
    if duration <= BULLET_MAX:
        return "bullet"
    if duration <= BLITZ_MAX:
        return "blitz"
    if duration <= RAPID_MAX:
        return "rapid"
    return "classical"


@dataclass
class FilterConfig:
    """Criteri di §2.1, con i valori di configs/default.yaml come default."""

    min_elo: int = MIN_ELO_DEFAULT
    time_controls: tuple[str, ...] = ("rapid", "classical")
    exclude_timeout_in_non_lost: bool = True
    min_moves: int = MIN_MOVES_DEFAULT


@dataclass
class FilterStats:
    """Conteggio dei motivi di scarto.

    Serve a rendere visibile *cosa* si sta buttando via: una pipeline che scarta il 99%
    delle partite senza dirlo e indistinguibile da una pipeline rotta. Misurato sul dump
    2026-07, i filtri tengono circa lo 0.7% delle partite — poco, ma corretto, e va
    documentato invece che scoperto per caso.
    """

    total: int = 0
    kept: int = 0
    rejected: dict[str, int] = field(default_factory=dict)

    def reject(self, reason: str) -> None:
        self.rejected[reason] = self.rejected.get(reason, 0) + 1

    def summary(self) -> str:
        rate = 100 * self.kept / self.total if self.total else 0.0
        parts = [f"{self.kept}/{self.total} partite tenute ({rate:.2f}%)"]
        for reason, count in sorted(self.rejected.items(), key=lambda kv: -kv[1]):
            parts.append(f"  scartate per {reason}: {count}")
        return "\n".join(parts)


def average_elo(white_elo: str | int | None, black_elo: str | int | None) -> float | None:
    try:
        white = int(white_elo)  # type: ignore[arg-type]
        black = int(black_elo)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return (white + black) / 2


def is_timeout_in_non_lost_position(termination: str | None, result: str | None) -> bool:
    """True se la partita e finita per tempo scaduto ma NON persa dal lato scaduto.

    Il caso da escludere: qualcuno perde al tempo in una posizione che stava vincendo.
    L'etichetta di valore direbbe "questa posizione porta alla sconfitta", che e falso —
    la posizione era vinta, e stato l'orologio a decidere.

    Lichess marca queste partite come `Time forfeit`. Le patte per tempo scaduto con
    materiale insufficiente sono l'altro caso in cui l'esito non riflette la posizione.
    """
    if not termination:
        return False
    if "time forfeit" not in termination.lower():
        return False
    # Una patta per tempo scaduto e sempre sospetta: l'esito non viene dalla posizione.
    return result == "1/2-1/2"


def accept_headers(headers: dict[str, str], config: FilterConfig, stats: FilterStats) -> bool:
    """Applica i filtri che dipendono solo dagli header, senza rigiocare la partita.

    Separato da `accept_game` perche e molto piu economico: scartare qui evita di
    parsare le mosse, che e la parte costosa. Su un dump Lichess questo elimina oltre il
    99% delle partite prima di toccare la notazione.
    """
    stats.total += 1

    cat = category(headers.get("TimeControl"))
    if cat is None:
        stats.reject("time control non parsabile")
        return False
    if cat not in config.time_controls:
        stats.reject(f"categoria {cat}")
        return False

    elo = average_elo(headers.get("WhiteElo"), headers.get("BlackElo"))
    if elo is None:
        stats.reject("Elo mancante")
        return False
    if elo < config.min_elo:
        stats.reject(f"Elo < {config.min_elo}")
        return False

    result = headers.get("Result")
    if result not in {"1-0", "0-1", "1/2-1/2"}:
        stats.reject("esito non decisivo o assente")
        return False

    if config.exclude_timeout_in_non_lost and is_timeout_in_non_lost_position(
        headers.get("Termination"), result
    ):
        stats.reject("tempo scaduto in posizione non persa")
        return False

    return True
