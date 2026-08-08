"""Valutazione materiale + piece-square tables (Fase 1 del piano).

Semplice di proposito: la baseline serve come test di sanita dell'intero progetto, non
come motore forte. Se la rete neurale non batte questa, c'e un bug — non un limite del
modello.

> Criticita #2, prima occorrenza — La valutazione deve essere ANTISIMMETRICA:
> `evaluate(pos) == -evaluate(pos a colori invertiti)`. Un segno sbagliato qui non
> crasha: produce un motore che gioca bene con un colore e male con l'altro. Il test
> di simmetria e nel Gate 2, e la stessa proprieta tornera nel backup MCTS (Gate 5).
"""

from __future__ import annotations

import chess

# Valori materiali in centipedoni (§Fase 1: P=1, N=B=3, R=5, Q=9).
# Il re non ha valore materiale: la sua perdita e gestita come matto, non come punteggio.
PIECE_VALUES: dict[int, int] = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

# Punteggio attribuito al matto. Volutamente sotto il limite degli interi usati come
# infinito nella ricerca, cosi "matto fra N mosse" resta ordinabile.
MATE_SCORE = 100_000

# --------------------------------------------------------------------------------------
# Piece-square tables
#
# Sono scritte dal punto di vista del BIANCO e con la prima riga = traversa 8, cioe
# nell'ordine in cui si legge una scacchiera stampata. `_flip_table` le converte
# nell'indicizzazione di python-chess (a1 = 0), e per il Nero si specchiano.
# I valori sono quelli classici di Tomasz Michniewski, usati da innumerevoli motori
# didattici: non sono tunati, e non devono esserlo — la baseline e un riferimento fisso.
# --------------------------------------------------------------------------------------

_PAWN_TABLE = [
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
]  # fmt: skip

_KNIGHT_TABLE = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
]  # fmt: skip

_BISHOP_TABLE = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
]  # fmt: skip

_ROOK_TABLE = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0,
]  # fmt: skip

_QUEEN_TABLE = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
]  # fmt: skip

# Il re ha due tabelle: in apertura sta al riparo dietro l'arrocco, in finale deve
# scendere in centro. Interpolare fra le due e la piu piccola aggiunta che evita il
# tipico finale in cui la baseline lascia il re nell'angolo.
_KING_MIDGAME_TABLE = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20,
]  # fmt: skip

_KING_ENDGAME_TABLE = [
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50,
]  # fmt: skip


def _flip_table(table: list[int]) -> tuple[int, ...]:
    """Da "prima riga = traversa 8" all'indicizzazione di python-chess (a1 = 0)."""
    flipped: list[int] = []
    for rank in range(8):
        start = (7 - rank) * 8
        flipped.extend(table[start : start + 8])
    return tuple(flipped)


PIECE_SQUARE_TABLES: dict[int, tuple[int, ...]] = {
    chess.PAWN: _flip_table(_PAWN_TABLE),
    chess.KNIGHT: _flip_table(_KNIGHT_TABLE),
    chess.BISHOP: _flip_table(_BISHOP_TABLE),
    chess.ROOK: _flip_table(_ROOK_TABLE),
    chess.QUEEN: _flip_table(_QUEEN_TABLE),
}

KING_TABLE_MIDGAME = _flip_table(_KING_MIDGAME_TABLE)
KING_TABLE_ENDGAME = _flip_table(_KING_ENDGAME_TABLE)

# Materiale non pedonale totale a inizio partita, per lato: 2N + 2B + 2R + Q.
_INITIAL_NON_PAWN = 2 * (320 + 330 + 500) + 900


def _endgame_weight(board: chess.Board) -> float:
    """0.0 in apertura, 1.0 a finale inoltrato, in base al materiale non pedonale."""
    remaining = 0
    for piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        count = len(board.pieces(piece_type, chess.WHITE)) + len(
            board.pieces(piece_type, chess.BLACK)
        )
        remaining += count * PIECE_VALUES[piece_type]
    total = 2 * _INITIAL_NON_PAWN
    return 1.0 - min(remaining / total, 1.0)


# Distanza di Chebyshev dal centro, per ogni casella: 0 al centro, 3 negli angoli.
# Precalcolata perche il termine di mate drive la interroga ad ogni valutazione.
_CENTER_DISTANCE: tuple[int, ...] = tuple(
    max(abs(chess.square_file(sq) * 2 - 7), abs(chess.square_rank(sq) * 2 - 7)) // 2
    for sq in range(64)
)


def _mate_drive(board: chess.Board, endgame: float) -> int:
    """Spinge verso il matto quando un lato ha un vantaggio decisivo senza pedoni.

    Senza questo termine la baseline vince materiale e poi non conclude: con alfiere e
    cavallo contro il re nudo sposta i pezzi a caso finche non scatta la regola delle 50
    mosse. E il classico "vinta ma patta", che rende la baseline inutile come metro di
    paragone — batterla diventa facile per ragioni sbagliate.

    Due incentivi, entrambi standard nei motori didattici:
    - spingere il re avversario verso il bordo (dove il matto e possibile)
    - avvicinare il proprio re, perche nei finali elementari serve per il matto
    """
    if endgame < 0.5:
        return 0

    white_material = sum(
        len(board.pieces(pt, chess.WHITE)) * PIECE_VALUES[pt]
        for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
    )
    black_material = sum(
        len(board.pieces(pt, chess.BLACK)) * PIECE_VALUES[pt]
        for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
    )

    # Serve un vantaggio da finale vinto, non un vantaggio qualsiasi: sotto una torre
    # di margine il termine farebbe piu danni che altro.
    diff = white_material - black_material
    if abs(diff) < PIECE_VALUES[chess.ROOK]:
        return 0

    strong, weak = (chess.WHITE, chess.BLACK) if diff > 0 else (chess.BLACK, chess.WHITE)
    strong_king = board.king(strong)
    weak_king = board.king(weak)
    if strong_king is None or weak_king is None:
        return 0

    # Re debole verso il bordo.
    score = 12 * _CENTER_DISTANCE[weak_king]
    # Re forte vicino a quello debole: 14 - distanza, cosi avvicinarsi paga.
    score += 4 * (14 - chess.square_distance(strong_king, weak_king))

    scaled = round(score * endgame)
    return scaled if strong == chess.WHITE else -scaled


def evaluate(board: chess.Board) -> int:
    """Valuta la posizione in centipedoni, dal punto di vista di CHI DEVE MUOVERE.

    Positivo significa "sto meglio io che devo muovere", indipendentemente dal colore.
    E la convenzione negamax, la stessa che usera il backup MCTS: tenerla uniforme fra
    baseline e rete e cio che rende confrontabili le due valutazioni.

    > Criticita #2 — Da questa convenzione discende l'antisimmetria:
    > `evaluate(b) == -evaluate(b.mirror())`. Il Gate 2 la verifica esplicitamente.
    """
    if board.is_checkmate():
        # Chi deve muovere e sotto matto: ha perso.
        return -MATE_SCORE
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    endgame = _endgame_weight(board)
    score = 0

    for square, piece in board.piece_map().items():
        # Le tabelle sono scritte per il Bianco: per il Nero si specchia la casella.
        table_square = square if piece.color == chess.WHITE else chess.square_mirror(square)

        if piece.piece_type == chess.KING:
            positional = round(
                KING_TABLE_MIDGAME[table_square] * (1.0 - endgame)
                + KING_TABLE_ENDGAME[table_square] * endgame
            )
        else:
            positional = PIECE_SQUARE_TABLES[piece.piece_type][table_square]

        value = PIECE_VALUES[piece.piece_type] + positional
        score += value if piece.color == chess.WHITE else -value

    score += _mate_drive(board, endgame)

    # Fin qui il punteggio e dal punto di vista del Bianco: si converte a negamax.
    return score if board.turn == chess.WHITE else -score
