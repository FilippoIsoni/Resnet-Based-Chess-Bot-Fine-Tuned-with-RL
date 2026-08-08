"""Minimax con alpha-beta — la baseline della Fase 1.

Negamax a profondita fissa (3-4 di default), con quiescence search sulle catture e
ordinamento MVV-LVA. Circa 1200-1400 Elo attesi.

La quiescence non e un vezzo: senza di essa un motore a profondita fissa soffre di
effetto orizzonte e regala pezzi in modo vistoso, il che renderebbe la baseline inutile
come metro di paragone (§Fase 1: se la rete non batte questa, c'e un bug).

> Criticita #2 — La convenzione e negamax: ogni valutazione e dal punto di vista di chi
> deve muovere, e ad ogni livello il valore del figlio viene NEGATO. Un segno sbagliato
> qui e lo stesso bug che tornera nel backup MCTS, dove sara molto piu difficile da
> vedere. Lo intercettano i test di simmetria in `tests/unit/test_baseline_evaluation.py`
> e il match contro l'avversario casuale.

> Nota sui punteggi potati — Il valore che alpha-beta restituisce per un ramo potato non
> e il valore di quel ramo: e il limite a cui la potatura si e fermata. Usarlo per
> qualcosa di diverso dallo scegliere il massimo (ordinare, campionare, confrontare
> mosse fra loro) produce numeri sbagliati. E costato una baseline che pattava contro
> l'avversario casuale: vedi docs/JOURNAL.md, 2026-08-08.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

import chess

from .evaluation import MATE_SCORE, PIECE_VALUES, evaluate

# Oltre il punteggio di matto: usato come infinito nella finestra alpha-beta.
INFINITY = MATE_SCORE * 10

DEFAULT_DEPTH = 4
DEFAULT_QUIESCENCE_DEPTH = 6

# Penalita applicata alla radice a una mossa che ripete la posizione o avvicina la
# regola delle 50 mosse, quando si sta vincendo. Deve superare il rumore posizionale
# delle PST (poche decine di centipedoni) senza avvicinarsi al valore di un pezzo.
REPETITION_PENALTY = 150


@dataclass
class SearchStats:
    """Contatori di una ricerca, utili per il profiling (regola #4)."""

    nodes: int = 0
    quiescence_nodes: int = 0
    cutoffs: int = 0
    elapsed: float = 0.0

    @property
    def nps(self) -> float:
        total = self.nodes + self.quiescence_nodes
        return total / self.elapsed if self.elapsed > 0 else 0.0


@dataclass
class SearchResult:
    move: chess.Move | None
    score: int
    depth: int
    stats: SearchStats = field(default_factory=SearchStats)

    @property
    def is_mate(self) -> bool:
        return abs(self.score) >= MATE_SCORE - 1000


def _mvv_lva(board: chess.Board, move: chess.Move) -> int:
    """Ordinamento delle mosse: prima le catture di pezzi grossi con pezzi piccoli.

    Un buon ordinamento non cambia il risultato della ricerca, cambia quanto costa
    ottenerlo: con alpha-beta la differenza fra ordinamento buono e casuale e di un
    ordine di grandezza nei nodi visitati.
    """
    score = 0
    if board.is_capture(move):
        victim = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)
        # En passant: la casella di arrivo e vuota ma la cattura c'e.
        victim_value = PIECE_VALUES[victim.piece_type] if victim else PIECE_VALUES[chess.PAWN]
        attacker_value = PIECE_VALUES[attacker.piece_type] if attacker else 0
        score += 10_000 + victim_value * 10 - attacker_value
    if move.promotion:
        score += 9_000 + PIECE_VALUES.get(move.promotion, 0)
    if board.gives_check(move):
        score += 500
    return score


def _ordered_moves(board: chess.Board) -> list[chess.Move]:
    return sorted(board.legal_moves, key=lambda m: -_mvv_lva(board, m))


def _quiescence(board: chess.Board, alpha: int, beta: int, depth: int, stats: SearchStats) -> int:
    """Estende la ricerca sulle sole catture, per non fermarsi a meta scambio."""
    stats.quiescence_nodes += 1

    stand_pat = evaluate(board)
    if depth == 0:
        return stand_pat
    if stand_pat >= beta:
        return beta
    alpha = max(alpha, stand_pat)

    for move in _ordered_moves(board):
        if not board.is_capture(move) and not move.promotion:
            continue
        board.push(move)
        score = -_quiescence(board, -beta, -alpha, depth - 1, stats)
        board.pop()

        if score >= beta:
            stats.cutoffs += 1
            return beta
        alpha = max(alpha, score)

    return alpha


def _negamax(
    board: chess.Board,
    depth: int,
    alpha: int,
    beta: int,
    stats: SearchStats,
    quiescence_depth: int,
    ply: int = 0,
) -> int:
    stats.nodes += 1

    if board.is_checkmate():
        # `ply` rende un matto vicino preferibile a uno lontano: senza, il motore
        # trova il matto ma puo rimandarlo all'infinito.
        return -MATE_SCORE + ply
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
    if board.can_claim_fifty_moves():
        return 0
    if ply > 0 and board.is_repetition(2):
        # Gia vista una volta nella partita o nella linea: trattarla come patta
        # IMMEDIATA, senza aspettare la terza occorrenza. E la convenzione standard
        # dei motori (Stockfish compreso): la seconda occorrenza basta, perche se una
        # linea porta a ripetere, l'avversario puo sempre insistere fino alla terza.
        #
        # Senza questo, con KR vs K la baseline muoveva la torre avanti e indietro
        # fino alla patta: la ripetizione era invisibile alla ricerca, che vedeva
        # tutte le mosse equivalenti.
        return 0

    if depth == 0:
        return _quiescence(board, alpha, beta, quiescence_depth, stats)

    best = -INFINITY
    for move in _ordered_moves(board):
        board.push(move)
        score = -_negamax(board, depth - 1, -beta, -alpha, stats, quiescence_depth, ply + 1)
        board.pop()

        best = max(best, score)
        alpha = max(alpha, score)
        if alpha >= beta:
            stats.cutoffs += 1
            break

    return best


def search(
    board: chess.Board,
    depth: int = DEFAULT_DEPTH,
    *,
    quiescence_depth: int = DEFAULT_QUIESCENCE_DEPTH,
    rng: random.Random | None = None,
) -> SearchResult:
    """Cerca la mossa migliore alla profondita data.

    Args:
        board: posizione di partenza. Non viene modificata (si lavora su una copia).
        depth: profondita in semimosse.
        quiescence_depth: estensione massima sulle catture.
        rng: se fornito, le mosse di pari punteggio vengono scelte a caso invece che
            in ordine di generazione. Serve nei match: senza, due baseline identiche
            giocano sempre la stessa partita e il risultato non ha valore statistico.

    Returns:
        `SearchResult` con la mossa, il punteggio dal punto di vista di chi muove e le
        statistiche della ricerca.
    """
    if depth < 1:
        raise ValueError(f"profondita minima 1, ricevuta {depth}")

    work = board.copy()
    stats = SearchStats()
    started = time.perf_counter()

    best_score = -INFINITY
    best_moves: list[chess.Move] = []

    for move in _ordered_moves(work):
        work.push(move)
        # Finestra PIENA ad ogni mossa della radice, non `(-INFINITY, -alpha)`.
        #
        # Restringere la finestra con alpha e la cosa giusta da fare quando serve solo
        # sapere QUALE mossa e la migliore: i rami peggiori vengono potati e la ricerca
        # costa molto meno. Ma il punteggio che tornano non e il loro valore vero — e
        # il limite a cui la potatura si e fermata. Qui i punteggi servono anche per
        # raccogliere i pari-merito in `best_moves`, e con la finestra stretta mosse
        # che valgono -942 tornavano +125 (il valore di cutoff), finendo nel gruppo
        # delle "migliori" ed estratte a caso da `rng`.
        #
        # Sintomo: la baseline perdeva o pattava contro l'avversario CASUALE, senza mai
        # regalare un pezzo in modo vistoso — sceglieva a caso fra mosse che credeva
        # equivalenti. Costo della finestra piena: circa 2-3x nei nodi visitati, che a
        # questa profondita e accettabile. La potatura vera resta dentro `_negamax`.
        score = -_negamax(work, depth - 1, -INFINITY, INFINITY, stats, quiescence_depth, ply=1)

        # Il controllo dentro `_negamax` valuta 0 le ripetizioni raggiunte NELL'albero,
        # ma la mossa scelta alla radice puo ripetere comunque quando tutte le
        # alternative sembrano equivalenti. Qui si rompe il pareggio a sfavore della
        # ripetizione, cosi il motore preferisce progredire quando sta vincendo.
        if score > 0 and (work.is_repetition(2) or work.can_claim_fifty_moves()):
            score -= REPETITION_PENALTY
        work.pop()

        if score > best_score:
            best_score = score
            best_moves = [move]
        elif score == best_score:
            best_moves.append(move)

    stats.elapsed = time.perf_counter() - started

    if not best_moves:
        return SearchResult(None, evaluate(work), depth, stats)

    chosen = rng.choice(best_moves) if rng is not None else best_moves[0]
    return SearchResult(chosen, best_score, depth, stats)


def best_move(
    board: chess.Board, depth: int = DEFAULT_DEPTH, *, rng: random.Random | None = None
) -> chess.Move | None:
    """Scorciatoia: la sola mossa, senza statistiche."""
    return search(board, depth, rng=rng).move
