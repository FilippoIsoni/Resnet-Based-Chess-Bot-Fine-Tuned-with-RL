"""Self-play con batching FRA PARTITE (§5.4 del piano, criticita #8).

> La Fase 5 e il 90% del costo del progetto, e quasi tutto e generazione partite.

## Il problema

Batchare le foglie di **un** albero (§4.3, gia fatto) porta batch da ~20 posizioni: e il
limite naturale della larghezza dell'albero, oltre il quale le simulazioni collidono fra
loro. Su una GPU che digerisce 512 posizioni in un colpo, significa lavorare al 4%.

## La soluzione

Giocare **96 partite in parallelo** nello stesso processo, e raccogliere le foglie di
tutte prima di chiamare la rete. Ogni partita ha il suo albero, ma la valutazione e
condivisa: 96 alberi x ~20 foglie = batch da centinaia.

    una partita per volta:     [20 foglie] -> GPU -> [20 valori]
    96 partite in parallelo:   [~500 foglie da 96 alberi] -> GPU -> [~500 valori]

Guadagno atteso dal piano: **5-10x**. E il criterio del Gate 6 di ingresso.

## Come funziona il ciclo

Le partite avanzano **a passo di simulazione**, non a passo di mossa:

    finche ci sono partite vive:
        per ogni partita: scendi nell'albero, raccogli una foglia
        valuta TUTTE le foglie in un colpo
        per ogni partita: espandi e fai il backup
        ogni `simulations` giri: ogni partita gioca la sua mossa e avanza

Le partite finiscono in momenti diversi: quelle concluse escono dal gruppo e le altre
proseguono, con batch che si assottigliano verso la fine.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import chess
import numpy as np

from chessbot.search.mcts import SearchConfig, add_dirichlet_noise, expand, select_move
from chessbot.search.node import (
    Node,
    add_virtual_loss,
    backup,
    remove_virtual_loss,
    select_child,
    terminal_value,
)


@dataclass
class GameRecord:
    """Una posizione di self-play, con le etichette per il training.

    E cio che l'Expert Iteration produce e su cui la rete si riallena: la distribuzione
    delle visite al posto della mossa umana, e un target del valore che mescola il
    risultato con la stima della ricerca (§5.5).
    """

    fen: str
    visit_counts: dict[chess.Move, int]
    """La distribuzione π: quale mossa la ricerca ha scelto e con quanta convinzione.

    E il target della policy. E dimostrabilmente migliore della policy della rete,
    perche e quella policy **piu** N simulazioni di verifica: l'MCTS e un operatore di
    miglioramento garantito (§5.1)."""

    q_root: float
    """Q alla radice: la stima del valore prodotta dalla ricerca appena fatta, dal punto
    di vista di chi muove. Molto meno rumorosa del risultato finale (§5.5)."""

    to_play: chess.Color
    result: float | None = None
    """z: esito della partita dal punto di vista di chi muoveva. Riempito a fine partita."""


@dataclass
class SelfPlayStats:
    games: int = 0
    positions: int = 0
    network_calls: int = 0
    positions_evaluated: int = 0
    results: dict[str, int] = field(default_factory=lambda: {"1-0": 0, "0-1": 0, "1/2-1/2": 0})
    elapsed: float = 0.0

    @property
    def avg_batch(self) -> float:
        return self.positions_evaluated / self.network_calls if self.network_calls else 0.0

    @property
    def draw_rate(self) -> float:
        total = sum(self.results.values())
        return self.results["1/2-1/2"] / total if total else 0.0

    @property
    def games_per_hour(self) -> float:
        return self.games / self.elapsed * 3600 if self.elapsed else 0.0


class _Game:
    """Lo stato di una partita in corso dentro il gruppo parallelo."""

    __slots__ = ("board", "config", "done", "moves_played", "records", "root", "sims_done")

    def __init__(self, board: chess.Board, config: SearchConfig) -> None:
        self.board = board
        self.config = config
        self.root: Node | None = None
        self.records: list[GameRecord] = []
        self.sims_done = 0
        self.moves_played = 0
        self.done = False


def _collect_leaf(
    game: _Game, c_puct: float, virtual_loss: int
) -> tuple[list[Node], chess.Board] | None:
    """Scende nell'albero di una partita e raccoglie una foglia da valutare.

    Restituisce None se la foglia era terminale (risolta senza rete) o se la selezione
    e finita su un nodo gia in attesa.
    """
    assert game.root is not None
    path = [game.root]
    board = game.board.copy()
    node = game.root

    while node.is_expanded:
        move, node = select_child(node, c_puct)
        board.push(move)
        path.append(node)

    value = terminal_value(board)
    if value is not None:
        backup(path, value)
        game.sims_done += 1
        return None

    add_virtual_loss(path, virtual_loss)
    return path, board


def self_play_batch(
    evaluator,
    *,
    n_games: int,
    config: SearchConfig,
    openings: list[chess.Board] | None = None,
    max_plies: int = 300,
    temperature_moves: int = 15,
    rng: np.random.Generator | None = None,
    on_game_end=None,
) -> tuple[list[GameRecord], SelfPlayStats]:
    """Gioca `n_games` partite in parallelo, batchando le valutazioni fra tutte.

    Args:
        openings: posizioni di partenza. Se date, ogni partita ne pesca una a caso —
            e la contromisura alle patte del self-play (criticita #7). Senza, tutte le
            partite partono dalla posizione iniziale e si somigliano troppo.
        temperature_moves: per le prime N mosse la scelta e campionata invece che
            argmax, per diversificare le partite (§5.2).

    Returns:
        `(record di tutte le posizioni, statistiche)`.
    """
    import time

    rng = rng or np.random.default_rng()
    stats = SelfPlayStats()
    started = time.perf_counter()

    games: list[_Game] = []
    for _ in range(n_games):
        board = openings[int(rng.integers(len(openings)))].copy() if openings else chess.Board()
        games.append(_Game(board, config))

    all_records: list[GameRecord] = []

    while any(not g.done for g in games):
        vivi = [g for g in games if not g.done]

        # --- espandi le radici che non lo sono ancora ---------------------------------
        da_espandere = [g for g in vivi if g.root is None]
        if da_espandere:
            results = evaluator.evaluate([g.board for g in da_espandere])
            stats.network_calls += 1
            stats.positions_evaluated += len(da_espandere)
            for game, (priors, _) in zip(da_espandere, results, strict=True):
                game.root = Node(prior=1.0, to_play=game.board.turn)
                expand(game.root, game.board, priors)
                add_dirichlet_noise(
                    game.root, config.dirichlet_alpha, config.dirichlet_epsilon, rng
                )
                if not game.root.children:
                    game.done = True

        # --- raccogli una foglia per ogni partita, poi valuta tutte insieme -----------
        pending: list[tuple[_Game, list[Node], chess.Board]] = []
        for game in vivi:
            if game.done or game.sims_done >= config.simulations:
                continue
            leaf = _collect_leaf(game, config.c_puct, config.virtual_loss)
            if leaf is not None:
                path, board = leaf
                pending.append((game, path, board))
                game.sims_done += 1

        if pending:
            # E QUI che sta il guadagno: un solo forward per decine di partite diverse.
            results = evaluator.evaluate([b for _, _, b in pending])
            stats.network_calls += 1
            stats.positions_evaluated += len(pending)

            for (_game, path, board), (priors, value) in zip(pending, results, strict=True):
                remove_virtual_loss(path, config.virtual_loss)
                expand(path[-1], board, priors)
                backup(path, value)

        # --- le partite che hanno finito le simulazioni giocano la loro mossa ---------
        for game in vivi:
            if game.done or game.sims_done < config.simulations:
                continue

            root = game.root
            assert root is not None

            # Temperatura alta all'inizio: diversifica le partite (§5.2, criticita #7).
            move_config = SearchConfig(
                simulations=config.simulations,
                c_puct=config.c_puct,
                temperature=1.0 if game.moves_played < temperature_moves else 0.0,
            )
            move = select_move(root, move_config, rng)
            if move is None:
                game.done = True
                continue

            # Q alla radice, dal pov di chi muove: e il target del valore di §5.5.
            q_root = root.value_sum / root.visits if root.visits else 0.0

            game.records.append(
                GameRecord(
                    fen=game.board.fen(),
                    visit_counts={m: c.visits for m, c in root.children.items()},
                    q_root=q_root,
                    to_play=game.board.turn,
                )
            )

            game.board.push(move)
            game.moves_played += 1
            game.root = None
            game.sims_done = 0

            if game.board.is_game_over(claim_draw=True) or game.board.ply() >= max_plies:
                game.done = True

    # --- assegna il risultato a ogni posizione ----------------------------------------
    for game in games:
        outcome = (
            game.board.result(claim_draw=True)
            if game.board.is_game_over(claim_draw=True)
            else "1/2-1/2"
        )
        stats.results[outcome] = stats.results.get(outcome, 0) + 1
        stats.games += 1

        # z dal punto di vista di CHI MUOVEVA in quella posizione: alterna ad ogni ply.
        z_white = {"1-0": 1.0, "0-1": -1.0, "1/2-1/2": 0.0}[outcome]
        for record in game.records:
            record.result = z_white if record.to_play == chess.WHITE else -z_white

        all_records.extend(game.records)
        stats.positions += len(game.records)
        if on_game_end:
            on_game_end(game.board, outcome)

    stats.elapsed = time.perf_counter() - started
    return all_records, stats


def value_target(record: GameRecord, alpha_q_root: float) -> float:
    """Target del valore: mescola la stima della ricerca con il risultato (§5.5).

        target = alpha * Q_radice + (1 - alpha) * z

    > `Q_radice` e una stima **cercata**, molto meno rumorosa del risultato finale: una
    > posizione vinta puo finire persa venti mosse dopo, e `z` non sa distinguere le due
    > cose. Costa una riga ed e particolarmente utile qui, dove la value head parte
    > rumorosa (criticita #5, misurata al Gate 4: 0,067 nats sulla baseline banale).
    """
    z = record.result if record.result is not None else 0.0
    return alpha_q_root * record.q_root + (1.0 - alpha_q_root) * z


__all__ = [
    "GameRecord",
    "SelfPlayStats",
    "self_play_batch",
    "value_target",
]
