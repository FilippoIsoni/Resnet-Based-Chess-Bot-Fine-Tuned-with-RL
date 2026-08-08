"""Encoder posizione -> tensore 19x8x8 (§2.3 del piano).

    piani  0-5    pedoni, cavalli, alfieri, torri, donne, re — LATO CHE MUOVE
    piani  6-11   stessi sei tipi — LATO AVVERSARIO
    piani 12-15   diritti di arrocco: mio corto, mio lungo, suo corto, suo lungo
    piano 16      colonna dell'en passant
    piano 17      contatore delle 50 mosse, normalizzato
    piano 18      conteggio delle ripetizioni della posizione

Nessun piano per il turno: la scacchiera e sempre orientata dal lato che muove, quindi
un piano "tocca a me" varrebbe costantemente 1 e non trasporterebbe informazione.

Non e un embedding: e una codifica deterministica decisa a mano, piu simile a un one-hot.

> Criticita #1 — Orientare la scacchiera dal lato che muove significa, quando tocca al
> Nero, specchiare verticalmente E scambiare i colori. L'indice della mossa-etichetta va
> specchiato con la stessa trasformazione: vedi `moves.mirror_move_index`. La funzione
> `encode_sample` fa entrambe le cose insieme, ed e il modo raccomandato di produrre
> esempi di training proprio per rendere impossibile dimenticarne una.
"""

from __future__ import annotations

import chess
import numpy as np

from .moves import mirror_move_index, move_to_index

N_PLANES = 19
BOARD_SIZE = 8

# Ordine dei tipi di pezzo nei piani 0-5 e 6-11. Fissato: cambiarlo invalida i modelli
# gia allenati e i file golden.
PIECE_ORDER: tuple[int, ...] = (
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
)

# Il contatore delle 50 mosse si normalizza su 100 semimosse, che e la soglia della
# patta. Il valore viene troncato a 1.0: oltre la soglia la partita e comunque finita.
HALFMOVE_NORMALIZER = 100.0

# Il conteggio ripetizioni si normalizza su 3, il numero che fa scattare la patta.
REPETITION_NORMALIZER = 3.0


def _oriented_board(board: chess.Board) -> chess.Board:
    """Restituisce la board vista da chi muove: il Bianco cosi com'e, il Nero specchiato.

    `chess.Board.mirror()` fa esattamente la trasformazione voluta — flip verticale piu
    scambio dei colori — e riporta il turno al Bianco. Dopo questa chiamata "il mio
    esercito" sale sempre, che e il punto dell'orientamento (§2.3).
    """
    return board if board.turn == chess.WHITE else board.mirror()


def encode_board(board: chess.Board, *, repetitions: int | None = None) -> np.ndarray:
    """Codifica una posizione nel tensore 19x8x8, orientato dal lato che muove.

    Args:
        board: la posizione. Non viene modificata.
        repetitions: quante volte la posizione si e gia presentata. Se None viene
            dedotto dalla board, che pero lo sa solo se possiede lo storico delle mosse:
            una board ricostruita da FEN riporta sempre 0. Nel preprocessing conviene
            passarlo esplicitamente, calcolandolo mentre si scorre la partita.

    Returns:
        Array float32 di forma (19, 8, 8). L'indice [p, r, f] e il piano p alla traversa
        r e colonna f della scacchiera ORIENTATA.
    """
    if repetitions is None:
        # is_repetition(n) e costoso ma corretto; su una board senza storico da 0.
        repetitions = 2 if board.is_repetition(3) else (1 if board.is_repetition(2) else 0)

    pov = _oriented_board(board)
    planes = np.zeros((N_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

    # Piani 0-11: i pezzi. Dopo l'orientamento chi muove e sempre il Bianco, quindi
    # "mio" == WHITE e "suo" == BLACK senza ulteriori condizionali.
    for color_offset, color in ((0, chess.WHITE), (6, chess.BLACK)):
        for i, piece_type in enumerate(PIECE_ORDER):
            plane = planes[color_offset + i]
            for square in pov.pieces(piece_type, color):
                plane[chess.square_rank(square), chess.square_file(square)] = 1.0

    # Piani 12-15: diritti di arrocco, come piani costanti 0/1.
    planes[12].fill(float(pov.has_kingside_castling_rights(chess.WHITE)))
    planes[13].fill(float(pov.has_queenside_castling_rights(chess.WHITE)))
    planes[14].fill(float(pov.has_kingside_castling_rights(chess.BLACK)))
    planes[15].fill(float(pov.has_queenside_castling_rights(chess.BLACK)))

    # Piano 16: colonna dell'en passant. Si marca l'intera colonna della casella di
    # presa al varco; nessuna casella accesa se non ce n'e una.
    if pov.ep_square is not None:
        planes[16][:, chess.square_file(pov.ep_square)] = 1.0

    # Piani 17-18: contatori scalari, spalmati su tutto il piano.
    planes[17].fill(min(board.halfmove_clock / HALFMOVE_NORMALIZER, 1.0))
    planes[18].fill(min(repetitions / REPETITION_NORMALIZER, 1.0))

    return planes


def encode_move(board: chess.Board, move: chess.Move) -> int:
    """Codifica una mossa nell'indice 0-4671 COERENTE con `encode_board`.

    > Criticita #1 — questa e la meta che si dimentica. Se la board e stata orientata
    > (tocca al Nero), l'indice va specchiato. Usare `moves.move_to_index` direttamente
    > su una posizione del Nero produce un'etichetta sbagliata, senza errori a runtime.
    """
    index = move_to_index(move)
    return index if board.turn == chess.WHITE else mirror_move_index(index)


def decode_move(board: chess.Board, index: int) -> chess.Move:
    """Inverte `encode_move`: dall'indice orientato alla mossa nella board originale."""
    if board.turn == chess.BLACK:
        index = mirror_move_index(index)
        # index_to_move ha bisogno di sapere in che verso avanzano i pedoni e se una
        # mossa e una promozione: glielo dice la board vera, non quella orientata.
    return _index_to_move_on(board, index)


def _index_to_move_on(board: chess.Board, index: int) -> chess.Move:
    from .moves import index_to_move

    return index_to_move(index, board)


def encode_sample(
    board: chess.Board, move: chess.Move, *, repetitions: int | None = None
) -> tuple[np.ndarray, int]:
    """Produce la coppia (tensore, indice mossa) gia coerentemente orientata.

    E il modo raccomandato di costruire esempi di training: tenere insieme le due meta
    rende strutturalmente impossibile flippare la scacchiera dimenticando l'etichetta
    (criticita #1).
    """
    return encode_board(board, repetitions=repetitions), encode_move(board, move)


def legal_move_mask(board: chess.Board) -> np.ndarray:
    """Maschera booleana di forma (4672,): True sulle mosse legali, orientata come l'input.

    La stessa maschera deve essere usata in training e in inferenza. Applicarla solo da
    una parte e un bug silenzioso: la rete impara a distribuire probabilita su mosse che
    poi non le vengono mai proposte (Gate 4).
    """
    from .moves import N_MOVES

    mask = np.zeros(N_MOVES, dtype=bool)
    for move in board.legal_moves:
        mask[encode_move(board, move)] = True
    return mask


def decode_planes_to_fen(planes: np.ndarray, *, side_to_move: chess.Color) -> str:
    """Ricostruisce la FEN dai piani — verifica incrociata indipendente (§2.5).

    Serve a dimostrare che l'encoder non perde informazione: si ricostruisce la
    posizione dal tensore e la si confronta con `board.fen()`. Il confronto e possibile
    solo sui campi che il tensore contiene davvero, quindi il numero di mossa e omesso.

    Args:
        planes: tensore (19, 8, 8) prodotto da `encode_board`.
        side_to_move: chi muoveva nella posizione originale. Non e nel tensore di
            proposito (§2.3), quindi va fornito da fuori per poter de-orientare.

    Returns:
        FEN dei primi quattro campi piu l'halfmove clock, con il numero di mossa a 1.
    """
    if planes.shape != (N_PLANES, BOARD_SIZE, BOARD_SIZE):
        raise ValueError(f"forma attesa (19, 8, 8), ricevuta {planes.shape}")

    board = chess.Board.empty()
    for color_offset, color in ((0, chess.WHITE), (6, chess.BLACK)):
        for i, piece_type in enumerate(PIECE_ORDER):
            for rank, file in zip(*np.nonzero(planes[color_offset + i])):
                board.set_piece_at(
                    chess.square(int(file), int(rank)), chess.Piece(piece_type, color)
                )

    castling = ""
    if planes[12][0, 0]:
        castling += "K"
    if planes[13][0, 0]:
        castling += "Q"
    if planes[14][0, 0]:
        castling += "k"
    if planes[15][0, 0]:
        castling += "q"
    board.set_castling_fen(castling or "-")

    ep_files = np.nonzero(planes[16].any(axis=0))[0]
    if len(ep_files):
        # Nella board orientata chi muove e sempre il Bianco, quindi la casella di presa
        # al varco sta sulla sesta traversa.
        board.ep_square = chess.square(int(ep_files[0]), 5)

    board.halfmove_clock = round(float(planes[17][0, 0]) * HALFMOVE_NORMALIZER)
    board.turn = chess.WHITE

    # Se muoveva il Nero il tensore descrive la posizione specchiata: si de-orienta
    # applicando di nuovo il mirror, che e involutivo.
    if side_to_move == chess.BLACK:
        board = board.mirror()

    # `Board.fen()` omette l'en passant quando nessun pedone puo effettivamente
    # catturare al varco, mentre `shredder_fen()`/il campo grezzo lo conservano. Qui
    # serve il campo grezzo: il confronto deve essere fra cio che il tensore contiene e
    # cio che la posizione di partenza dichiarava, non fra due normalizzazioni diverse.
    return board.fen(en_passant="fen")
