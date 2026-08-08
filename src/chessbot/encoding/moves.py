"""Codifica delle mosse in indici 0-4671 (§2.4 del piano).

Schema AlphaZero: 8x8x73 = 4672 indici. Per ogni casella di partenza, 73 "tipi di mossa":

    tipi  0-55   mosse "da regina": 8 direzioni x 7 distanze
    tipi 56-63   mosse di cavallo
    tipi 64-72   sottopromozioni: 3 pezzi (cavallo, alfiere, torre) x 3 direzioni

L'indice e `from_square * 73 + move_type`, con `from_square` in convenzione python-chess
(a1 = 0, h8 = 63).

> Criticita #4 — La promozione a DONNA non sta nei piani di sottopromozione.
> Si codifica come una normale mossa da regina di un passo in avanti (o in diagonale se
> e una cattura). Confonderla con le sottopromozioni rompe la biiettivita: due mosse
> diverse finirebbero sullo stesso indice.

> Criticita #1 — Se la scacchiera viene orientata dal lato che muove (vedi `board.py`),
> l'indice della mossa va specchiato con la stessa trasformazione. Farne solo una delle
> due significa allenare su etichette sbagliate per meta del dataset, senza errori a
> runtime. Il flip vive qui, in `mirror_move_index`.
"""

from __future__ import annotations

import chess

# Numero totale di indici. Coincide con `encoding.n_moves` in configs/default.yaml.
N_MOVE_TYPES = 73
N_MOVES = 64 * N_MOVE_TYPES  # 4672

# Le 8 direzioni delle mosse da regina, come (delta_file, delta_rank).
# L'ordine e arbitrario ma FISSATO: cambiarlo invalida ogni modello gia allenato
# e ogni file golden. Non riordinare.
QUEEN_DIRECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1),  # N
    (1, 1),  # NE
    (1, 0),  # E
    (1, -1),  # SE
    (0, -1),  # S
    (-1, -1),  # SW
    (-1, 0),  # W
    (-1, 1),  # NW
)

# Gli 8 salti del cavallo, anch'essi in ordine fissato.
KNIGHT_DELTAS: tuple[tuple[int, int], ...] = (
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
)

# I tre pezzi di sottopromozione, in ordine fissato. La donna e esclusa di proposito
# (criticita #4): la sua promozione e una mossa da regina.
UNDERPROMOTION_PIECES: tuple[int, ...] = (chess.KNIGHT, chess.BISHOP, chess.ROOK)

# Le tre direzioni di una promozione, come delta_file dal punto di vista del pedone
# che avanza: cattura a sinistra, avanti dritto, cattura a destra.
UNDERPROMOTION_FILE_DELTAS: tuple[int, ...] = (-1, 0, 1)

_QUEEN_DIR_INDEX = {d: i for i, d in enumerate(QUEEN_DIRECTIONS)}
_KNIGHT_DELTA_INDEX = {d: i for i, d in enumerate(KNIGHT_DELTAS)}


class MoveEncodingError(ValueError):
    """Mossa non rappresentabile, o indice fuori dallo spazio 0-4671."""


def _file_rank(square: int) -> tuple[int, int]:
    return chess.square_file(square), chess.square_rank(square)


def move_to_index(move: chess.Move) -> int:
    """Converte una mossa nel suo indice 0-4671.

    La mossa deve essere gia espressa nell'orientamento in cui verra usata: questa
    funzione non sa nulla di chi muove e non applica alcun flip. Per il flip vedi
    `mirror_move_index`.

    Raises:
        MoveEncodingError: se la mossa non e rappresentabile nello schema (per esempio
            un salto che non e ne una linea di regina ne un salto di cavallo).
    """
    from_file, from_rank = _file_rank(move.from_square)
    to_file, to_rank = _file_rank(move.to_square)
    df, dr = to_file - from_file, to_rank - from_rank

    # Sottopromozioni: hanno il loro blocco dedicato e vanno controllate per prime,
    # altrimenti verrebbero assorbite dalle mosse da regina di un passo.
    if move.promotion is not None and move.promotion != chess.QUEEN:
        try:
            piece_idx = UNDERPROMOTION_PIECES.index(move.promotion)
        except ValueError:
            raise MoveEncodingError(f"pezzo di promozione non valido: {move.promotion}") from None
        if df not in UNDERPROMOTION_FILE_DELTAS:
            raise MoveEncodingError(f"promozione con spostamento laterale illegale: {move}")
        # Il pedone avanza di una traversa: +1 per il Bianco, -1 per il Nero. Il segno
        # non entra nell'indice perche una promozione avviene sempre "in avanti" per chi
        # la gioca, e sulla scacchiera orientata e sempre verso l'alto.
        if abs(dr) != 1:
            raise MoveEncodingError(f"promozione che non avanza di una traversa: {move}")
        dir_idx = UNDERPROMOTION_FILE_DELTAS.index(df)
        move_type = 64 + piece_idx * 3 + dir_idx
        return move.from_square * N_MOVE_TYPES + move_type

    # Cavallo.
    if (df, dr) in _KNIGHT_DELTA_INDEX:
        move_type = 56 + _KNIGHT_DELTA_INDEX[(df, dr)]
        return move.from_square * N_MOVE_TYPES + move_type

    # Mosse da regina, comprese le promozioni a donna (criticita #4).
    if df == 0 and dr == 0:
        raise MoveEncodingError(f"mossa nulla: {move}")
    if df == 0 or dr == 0 or abs(df) == abs(dr):
        distance = max(abs(df), abs(dr))
        if distance > 7:
            raise MoveEncodingError(f"distanza fuori scala: {move}")
        step = (0 if df == 0 else df // abs(df), 0 if dr == 0 else dr // abs(dr))
        dir_idx = _QUEEN_DIR_INDEX[step]
        move_type = dir_idx * 7 + (distance - 1)
        return move.from_square * N_MOVE_TYPES + move_type

    raise MoveEncodingError(f"mossa non rappresentabile nello schema 8x8x73: {move}")


def index_to_move(index: int, board: chess.Board | None = None) -> chess.Move:
    """Inverte `move_to_index`.

    Args:
        index: indice 0-4671.
        board: se fornita, viene usata per disambiguare i due casi che l'indice da solo
            non determina — la promozione a donna (un pedone che raggiunge l'ultima
            traversa promuove sempre) e nient'altro. Senza board la mossa viene
            restituita senza promozione, che e corretto per ogni pezzo diverso dal pedone
            in procinto di promuovere.

    Raises:
        MoveEncodingError: se l'indice e fuori range o descrive una casella di arrivo
            fuori dalla scacchiera.
    """
    if not 0 <= index < N_MOVES:
        raise MoveEncodingError(f"indice fuori da [0, {N_MOVES - 1}]: {index}")

    from_square, move_type = divmod(index, N_MOVE_TYPES)
    from_file, from_rank = _file_rank(from_square)

    if move_type < 56:
        dir_idx, dist_idx = divmod(move_type, 7)
        df, dr = QUEEN_DIRECTIONS[dir_idx]
        distance = dist_idx + 1
        to_file, to_rank = from_file + df * distance, from_rank + dr * distance
        promotion = None
    elif move_type < 64:
        df, dr = KNIGHT_DELTAS[move_type - 56]
        to_file, to_rank = from_file + df, from_rank + dr
        promotion = None
    else:
        piece_idx, dir_idx = divmod(move_type - 64, 3)
        promotion = UNDERPROMOTION_PIECES[piece_idx]
        df = UNDERPROMOTION_FILE_DELTAS[dir_idx]
        # La direzione di avanzamento dipende da chi muove. Senza board si assume il
        # Bianco: e la convenzione giusta perche il dataset e sempre orientato dal lato
        # che muove, dove i pedoni salgono.
        dr = 1
        if board is not None and board.turn == chess.BLACK:
            dr = -1
        to_file, to_rank = from_file + df, from_rank + dr

    if not (0 <= to_file < 8 and 0 <= to_rank < 8):
        raise MoveEncodingError(
            f"indice {index} porta fuori dalla scacchiera (file {to_file}, rank {to_rank})"
        )

    to_square = chess.square(to_file, to_rank)

    # Promozione a donna: l'indice la codifica come mossa da regina, quindi va
    # ricostruita qui guardando il pezzo. Serve la board — senza, si restituisce la
    # mossa senza promozione.
    if promotion is None and board is not None:
        piece = board.piece_at(from_square)
        if piece is not None and piece.piece_type == chess.PAWN:
            last_rank = 7 if piece.color == chess.WHITE else 0
            if to_rank == last_rank:
                promotion = chess.QUEEN

    return chess.Move(from_square, to_square, promotion=promotion)


def mirror_move_index(index: int) -> int:
    """Specchia un indice mossa come `chess.square_mirror` specchia una casella.

    > Criticita #1 — questa funzione e il rimedio al bug silenzioso piu costoso del
    > progetto. Quando `encode_board` orienta la scacchiera dal lato che muove (flip
    > verticale + scambio colori), l'indice della mossa-etichetta DEVE passare di qui.

    Il flip e verticale (la traversa r diventa 7-r, la colonna resta): la casella si
    specchia con `chess.square_mirror`, e il tipo di mossa si specchia invertendo il
    segno di `delta_rank`. Le colonne non cambiano, quindi le sottopromozioni mantengono
    la loro direzione laterale.
    """
    if not 0 <= index < N_MOVES:
        raise MoveEncodingError(f"indice fuori da [0, {N_MOVES - 1}]: {index}")

    from_square, move_type = divmod(index, N_MOVE_TYPES)
    mirrored_from = chess.square_mirror(from_square)

    if move_type < 56:
        dir_idx, dist_idx = divmod(move_type, 7)
        df, dr = QUEEN_DIRECTIONS[dir_idx]
        mirrored_dir = _QUEEN_DIR_INDEX[(df, -dr)]
        mirrored_type = mirrored_dir * 7 + dist_idx
    elif move_type < 64:
        df, dr = KNIGHT_DELTAS[move_type - 56]
        mirrored_type = 56 + _KNIGHT_DELTA_INDEX[(df, -dr)]
    else:
        # Le sottopromozioni sono gia espresse dal punto di vista di chi muove: il
        # pedone avanza sempre "verso l'alto" nella board orientata, e la direzione
        # laterale (delta_file) non e toccata da un flip verticale. Il tipo resta.
        mirrored_type = move_type

    return mirrored_from * N_MOVE_TYPES + mirrored_type
