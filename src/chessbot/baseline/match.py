"""Match fra due giocatori e misura del risultato con intervallo di confidenza.

> Regola #1 di PIPELINE.md — Ogni numero riportato ha un intervallo di confidenza.
> Con 200 partite l'errore su una misura Elo e circa ±35. Una differenza di 20 Elo non
> e una differenza. Questo modulo restituisce SEMPRE l'intervallo insieme al punteggio,
> proprio per rendere scomodo riportare il numero da solo.

Serve dal Gate 2 (100/100 contro random) in poi, e verra riusato per il gating della
Expert Iteration.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass

import chess

# Un giocatore e una funzione posizione -> mossa. Cosi baseline, rete e MCTS sono
# intercambiabili in un match senza che questo modulo sappia nulla di loro.
Player = Callable[[chess.Board], chess.Move | None]

# Limite di semimosse oltre il quale la partita e dichiarata patta. Serve a impedire
# che una partita patologica blocchi un match.
DEFAULT_MAX_PLIES = 300


@dataclass
class MatchResult:
    wins: int = 0
    losses: int = 0
    draws: int = 0

    @property
    def games(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def score(self) -> float:
        """Punteggio nel senso scacchistico: vittoria 1, patta 0.5, sconfitta 0."""
        return self.wins + 0.5 * self.draws

    @property
    def score_rate(self) -> float:
        return self.score / self.games if self.games else 0.0

    @property
    def elo_diff(self) -> float:
        """Differenza Elo stimata dal punteggio.

        Infinita (per segno) su un cappotto: con 100/100 il modello logistico non ha
        un massimo finito, e va detto invece di restituire un numero inventato.
        """
        rate = self.score_rate
        if rate <= 0.0:
            return float("-inf")
        if rate >= 1.0:
            return float("inf")
        return -400.0 * math.log10(1.0 / rate - 1.0)

    @property
    def elo_margin(self) -> float:
        """Semiampiezza dell'intervallo di confidenza al 95% sulla stima Elo.

        Errore standard binomiale propagato attraverso la derivata della curva Elo.
        Approssimazione grossolana ai bordi (rate vicino a 0 o 1), che e esattamente
        dove la stima Elo non ha comunque senso.
        """
        n = self.games
        rate = self.score_rate
        if n == 0 or rate <= 0.0 or rate >= 1.0:
            return float("inf")
        se = math.sqrt(rate * (1.0 - rate) / n)
        derivative = 400.0 / (math.log(10.0) * rate * (1.0 - rate))
        return 1.96 * se * derivative

    def summary(self) -> str:
        """Riga pronta da incollare in docs/RESULTS.md, intervallo compreso."""
        base = f"{self.wins}W-{self.losses}L-{self.draws}D su {self.games} ({self.score_rate:.1%})"
        if math.isinf(self.elo_diff):
            return f"{base}, Elo non stimabile (risultato estremo)"
        return f"{base}, Elo {self.elo_diff:+.0f} ± {self.elo_margin:.0f} (95%)"


def play_game(
    white: Player, black: Player, *, max_plies: int = DEFAULT_MAX_PLIES
) -> tuple[str, chess.Board]:
    """Gioca una partita e restituisce (risultato, posizione finale).

    Il risultato e "1-0", "0-1" o "1/2-1/2". Un giocatore che non restituisce una mossa
    in una posizione non terminale perde: e un bug, e va reso visibile come sconfitta
    invece che nascosto come patta.
    """
    board = chess.Board()
    while not board.is_game_over(claim_draw=True):
        if board.ply() >= max_plies:
            return "1/2-1/2", board
        player = white if board.turn == chess.WHITE else black
        move = player(board)
        if move is None or move not in board.legal_moves:
            return "0-1" if board.turn == chess.WHITE else "1-0", board
        board.push(move)
    return board.result(claim_draw=True), board


def play_match(
    player: Player,
    opponent: Player,
    *,
    games: int = 100,
    alternate_colors: bool = True,
    max_plies: int = DEFAULT_MAX_PLIES,
) -> MatchResult:
    """Gioca un match e restituisce il risultato dal punto di vista di `player`.

    I colori si alternano di default: un match giocato sempre con lo stesso colore
    misura anche il vantaggio del tratto, non solo la differenza di forza.
    """
    result = MatchResult()
    for i in range(games):
        player_is_white = (i % 2 == 0) if alternate_colors else True
        white, black = (player, opponent) if player_is_white else (opponent, player)
        outcome, _ = play_game(white, black, max_plies=max_plies)

        if outcome == "1/2-1/2":
            result.draws += 1
        elif (outcome == "1-0") == player_is_white:
            result.wins += 1
        else:
            result.losses += 1
    return result


def random_player(rng: random.Random) -> Player:
    """Avversario di riferimento del Gate 2: sceglie una mossa legale a caso."""

    def play(board: chess.Board) -> chess.Move | None:
        moves = list(board.legal_moves)
        return rng.choice(moves) if moves else None

    return play
