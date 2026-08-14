"""Ricerca MCTS PUCT — il motore che trasforma la rete in un giocatore (Stadio 5).

    from chessbot.search import Evaluator, SearchConfig, run_mcts

    evaluator = Evaluator(rete, device)
    risultato = run_mcts(board, evaluator, SearchConfig(simulations=400))
    board.push(risultato.move)

Due moduli:

- `node.py`  l'albero, la formula PUCT, il backup con la negazione del segno
- `mcts.py`  le quattro fasi, il batching delle foglie, la scelta della mossa

> **Il bug da temere e il segno del valore nel backup** (criticita #2). Non crasha:
> produce un motore che gioca contro se stesso e sembra solo debole. Il test di
> monotonicita del Gate 5 — la forza deve crescere con le simulazioni — esiste per
> intercettarlo.
"""

from .mcts import (
    Evaluator,
    SearchConfig,
    SearchResult,
    SearchStats,
    add_dirichlet_noise,
    describe,
    expand,
    policy_target_from,
    principal_variation,
    run_mcts,
    select_move,
)
from .node import (
    TERMINAL_DRAW,
    TERMINAL_LOSS,
    Node,
    add_virtual_loss,
    backup,
    puct_score,
    remove_virtual_loss,
    select_child,
    terminal_value,
)

__all__ = [
    "TERMINAL_DRAW",
    "TERMINAL_LOSS",
    "Evaluator",
    "Node",
    "SearchConfig",
    "SearchResult",
    "SearchStats",
    "add_dirichlet_noise",
    "add_virtual_loss",
    "backup",
    "describe",
    "expand",
    "policy_target_from",
    "principal_variation",
    "puct_score",
    "remove_virtual_loss",
    "run_mcts",
    "select_child",
    "select_move",
    "terminal_value",
]
