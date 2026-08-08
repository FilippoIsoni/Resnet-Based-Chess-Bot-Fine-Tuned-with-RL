"""Baseline minimax — il test di sanita dell'intero progetto (Fase 1 del piano).

Non e un esercizio di riscaldamento: e il metro contro cui si misura la rete neurale.
Se la rete non batte questa baseline, c'e un bug — non un limite del modello.

    from chessbot.baseline import best_move
    mossa = best_move(board, depth=4)

Forza attesa: 1200-1400 Elo. Sa dare matto con materiale abbondante, ma non padroneggia
i finali elementari come KR vs K, dove arriva alla patta per la regola delle 50 mosse.
E un limite noto e accettabile a questo livello: vedi docs/RESULTS.md.
"""

from .evaluation import MATE_SCORE, PIECE_SQUARE_TABLES, PIECE_VALUES, evaluate
from .match import MatchResult, Player, play_game, play_match, random_player
from .perft import PERFT_SUITE, perft, perft_divide
from .search import DEFAULT_DEPTH, SearchResult, SearchStats, best_move, search

__all__ = [
    "DEFAULT_DEPTH",
    "MATE_SCORE",
    "PERFT_SUITE",
    "PIECE_SQUARE_TABLES",
    "PIECE_VALUES",
    "MatchResult",
    "Player",
    "SearchResult",
    "SearchStats",
    "best_move",
    "evaluate",
    "perft",
    "perft_divide",
    "play_game",
    "play_match",
    "random_player",
    "search",
]
