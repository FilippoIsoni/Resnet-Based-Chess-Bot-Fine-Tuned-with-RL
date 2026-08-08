"""Perft — conteggio dei nodi a profondita fissa (criterio del Gate 2).

Perft conta le posizioni foglia raggiungibili in N semimosse. I valori esatti per le
posizioni standard sono noti e pubblicati da decenni: confrontarcisi verifica la
generazione delle mosse contro una fonte completamente indipendente dal nostro codice.

Qui la generazione e quella di `python-chess`, quindi il perft non sta collaudando
codice nostro: sta collaudando che stiamo USANDO correttamente la libreria (regole
speciali comprese) e fissa un riferimento di prestazioni per la criticita #6, dove si
dovra decidere se la generazione mosse e davvero il collo di bottiglia dell'MCTS.
"""

from __future__ import annotations

import chess

# Posizioni standard del Chess Programming Wiki, con i conteggi esatti per profondita.
# Non modificare questi numeri: sono la verita di riferimento, non un output atteso
# del nostro codice.
PERFT_SUITE: tuple[tuple[str, str, tuple[int, ...]], ...] = (
    (
        "initial",
        chess.STARTING_FEN,
        (1, 20, 400, 8_902, 197_281, 4_865_609),
    ),
    (
        "kiwipete",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        (1, 48, 2_039, 97_862, 4_085_603),
    ),
    (
        "position_3",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        (1, 14, 191, 2_812, 43_238, 674_624),
    ),
    (
        "position_4",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        (1, 6, 264, 9_467, 422_333),
    ),
    (
        "position_5",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        (1, 44, 1_486, 62_379, 2_103_487),
    ),
    (
        "position_6",
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        (1, 46, 2_079, 89_890, 3_894_594),
    ),
)


def perft(board: chess.Board, depth: int) -> int:
    """Conta le posizioni foglia a profondita `depth`."""
    if depth == 0:
        return 1
    if depth == 1:
        # Bulk counting: a profondita 1 basta contare le mosse legali, senza spingerle.
        return board.legal_moves.count()

    total = 0
    for move in board.legal_moves:
        board.push(move)
        total += perft(board, depth - 1)
        board.pop()
    return total


def perft_divide(board: chess.Board, depth: int) -> dict[str, int]:
    """Perft suddiviso per mossa di primo livello — lo strumento con cui si debugga.

    Quando un perft non torna, il divide dice QUALE mossa porta al sottoalbero
    sbagliato: si scende ricorsivamente finche non si isola la posizione colpevole.
    """
    if depth < 1:
        raise ValueError("perft_divide richiede depth >= 1")
    result: dict[str, int] = {}
    for move in board.legal_moves:
        board.push(move)
        result[move.uci()] = perft(board, depth - 1)
        board.pop()
    return dict(sorted(result.items()))
