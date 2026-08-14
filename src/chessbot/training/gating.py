"""Gating: la rete nuova e davvero migliore? (§5.2 del piano).

> Il gating e l'unica cosa che impedisce di accorgersi dopo cinque giorni di essere
> peggiorati monotonicamente.

Dopo ogni iterazione di Expert Iteration, la rete allenata sfida quella corrente. Se non
vince abbastanza, **viene buttata** e si riprova con altri dati.

## Perche 200 partite e non 100

Con 100 partite il rumore su un 50/50 e circa ±5%: la soglia del 55% cadrebbe dentro il
rumore, e si promuoverebbero reti identiche o peggiori una volta su tre. Con 200 partite
l'errore scende a ±3,5% e la soglia diventa significativa.

E la regola #1 applicata a una decisione binaria: senza intervallo di confidenza, il
numero non dice nulla.

## SPRT: fermarsi appena si sa abbastanza

Il test sequenziale del rapporto di probabilita valuta dopo **ogni partita** se ci sono
gia prove sufficienti per decidere. In pratica:

- se la rete nuova e nettamente migliore, si accetta dopo ~60-80 partite
- se e nettamente peggiore, si rifiuta ancora prima
- solo nei casi ambigui si arriva a 200

Risparmia in media il 40-50% delle partite, che su decine di iterazioni sono ore.

Le due ipotesi:

    H0: la nuova rete vale elo0 (default 0, cioe non e migliore)
    H1: la nuova rete vale elo1 (default +25 Elo)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum


class SprtVerdict(str, Enum):
    ACCEPT = "accept"
    """H1: la rete nuova e migliore, si promuove."""

    REJECT = "reject"
    """H0: non c'e evidenza di miglioramento, si scarta."""

    CONTINUE = "continue"
    """Servono altre partite per decidere."""


def elo_to_score(elo: float) -> float:
    """Da differenza Elo a punteggio atteso, col modello logistico."""
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


@dataclass
class SprtTest:
    """Test sequenziale del rapporto di probabilita (§5.2).

    Si aggiorna partita per partita e dice quando smettere. I limiti derivano dagli
    errori di prima e seconda specie:

        A = log((1 - beta) / alpha)      soglia di accettazione
        B = log(beta / (1 - alpha))      soglia di rifiuto
    """

    elo0: float = 0.0
    elo1: float = 25.0
    alpha: float = 0.05
    beta: float = 0.05
    max_games: int = 200

    wins: int = 0
    losses: int = 0
    draws: int = 0
    llr: float = 0.0
    """Log-likelihood ratio accumulato: la quantita di evidenza raccolta finora."""

    history: list[float] = field(default_factory=list)

    @property
    def games(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def upper_bound(self) -> float:
        return math.log((1.0 - self.beta) / self.alpha)

    @property
    def lower_bound(self) -> float:
        return math.log(self.beta / (1.0 - self.alpha))

    @property
    def score_rate(self) -> float:
        return (self.wins + 0.5 * self.draws) / self.games if self.games else 0.0

    def update(self, outcome: float) -> SprtVerdict:
        """Registra una partita. `outcome` e 1.0 vittoria, 0.5 patta, 0.0 sconfitta."""
        if outcome == 1.0:
            self.wins += 1
        elif outcome == 0.0:
            self.losses += 1
        else:
            self.draws += 1

        self.llr = self._compute_llr()
        self.history.append(self.llr)
        return self.verdict()

    def _compute_llr(self) -> float:
        """Log-likelihood ratio del modello trinomiale (vittoria / patta / sconfitta).

        Si sommano i contributi delle singole partite:

            LLR = W * log(pw1/pw0) + L * log(pl1/pl0) + D * log(pd1/pd0)

        dove `pw`, `pl`, `pd` sono le probabilita dei tre esiti sotto le due ipotesi. La
        quota di patte si assume la stessa sotto H0 e H1 — dipende dalla forza assoluta
        dei due giocatori, non dalla loro differenza — quindi il suo termine si annulla e
        restano solo vittorie e sconfitte.

        > **Non usare l'approssimazione con la varianza campionaria.** Il primo tentativo
        > calcolava `n*((mean-p0)^2 - (mean-p1)^2)/(2*var)` con `var` stimata dai dati:
        > con una sola partita la varianza e zero, il clamp a 1e-6 faceva esplodere la
        > LLR a **17.313** contro una soglia di 2,94, e il test decideva a caso dopo la
        > prima partita. Misurato su quattro scenari a forza nota: tutti decisi dopo un
        > solo incontro, tre su quattro sbagliati.
        """
        if self.wins + self.losses == 0:
            # Solo patte: nessuna informazione sulla differenza di forza.
            return 0.0

        p0 = elo_to_score(self.elo0)
        p1 = elo_to_score(self.elo1)

        # Probabilita condizionate a "la partita e stata decisa": e li che sta il segnale.
        # Sotto H0 con elo0=0 valgono entrambe 0,5; sotto H1 sono sbilanciate.
        decided0 = p0
        decided1 = p1

        return self.wins * math.log(decided1 / decided0) + self.losses * math.log(
            (1 - decided1) / (1 - decided0)
        )

    def verdict(self) -> SprtVerdict:
        if self.llr >= self.upper_bound:
            return SprtVerdict.ACCEPT
        if self.llr <= self.lower_bound:
            return SprtVerdict.REJECT
        if self.games >= self.max_games:
            # Esaurite le partite senza evidenza: si decide col punteggio, che e la
            # regola del gating classico (§5.2: >= 55%).
            return SprtVerdict.ACCEPT if self.score_rate >= 0.55 else SprtVerdict.REJECT
        return SprtVerdict.CONTINUE

    def summary(self) -> str:
        elo, margin = self.elo_estimate()
        elo_txt = "non stimabile" if math.isinf(elo) else f"{elo:+.0f} ± {margin:.0f}"
        return (
            f"{self.wins}W-{self.losses}L-{self.draws}D su {self.games} "
            f"({100 * self.score_rate:.1f}%), Elo {elo_txt}, "
            f"LLR {self.llr:+.2f} [{self.lower_bound:.2f}, {self.upper_bound:.2f}]"
        )

    def elo_estimate(self) -> tuple[float, float]:
        """Stima Elo e semiampiezza dell'intervallo al 95% (regola #1)."""
        rate = self.score_rate
        if self.games == 0 or rate <= 0.0 or rate >= 1.0:
            return (float("inf") if rate >= 1.0 else float("-inf")), float("inf")
        elo = -400.0 * math.log10(1.0 / rate - 1.0)
        se = math.sqrt(rate * (1.0 - rate) / self.games)
        derivative = 400.0 / (math.log(10.0) * rate * (1.0 - rate))
        return elo, 1.96 * se * derivative


__all__ = ["SprtTest", "SprtVerdict", "elo_to_score"]
