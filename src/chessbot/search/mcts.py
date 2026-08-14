"""MCTS PUCT: la ricerca che trasforma la rete in un motore (§4.2-4.3 del piano).

> Non allena nulla. E *planning*: computa una mossa e butta via l'albero. Ma e il salto
> di Elo piu grande del progetto, e non costa un solo step di training.

## Una simulazione, in quattro fasi

    1. SELEZIONE   dalla radice, scendi scegliendo il PUCT piu alto, finche non
                   arrivi a un nodo mai espanso
    2. ESPANSIONE  crea i figli di quel nodo, uno per mossa legale
    3. VALUTAZIONE chiedi alla rete quanto vale la posizione
    4. BACKUP      risali aggiornando le statistiche, negando il valore ad ogni livello

Ripeti N volte (400 di default), poi **gioca la mossa piu visitata**.

## Perche la piu visitata e non quella col Q piu alto

Controintuitivo ma essenziale. Una mossa valutata benissimo **una volta sola** puo essere
fortuna: la rete ha sbagliato una stima e nessuno l'ha corretta. Una visitata 200 volte
su 400 ha retto un esame ripetuto, perche ad ogni passaggio la ricerca ha avuto la
possibilita di scoprire una confutazione e spostarsi altrove.

Il conteggio delle visite e una statistica robusta; il valore medio no.

## Niente rollout casuali

Il Monte Carlo classico gioca a caso fino alla fine della partita. **Negli scacchi non
funziona**: sono a morte improvvisa, e con la donna in piu tre mosse casuali possono
regalare il matto. Il rumore tattico distrugge il segnale posizionale.

E la value head a rendere praticabile la ricerca. Il "Monte Carlo" nel nome e un residuo
storico: quello che resta e la parte *tree search*.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import chess
import numpy as np
import torch

from chessbot.encoding import encode_board, legal_move_mask
from chessbot.model.network import masked_policy_logits
from chessbot.search.node import (
    Node,
    add_virtual_loss,
    backup,
    remove_virtual_loss,
    select_child,
    terminal_value,
)


@dataclass
class SearchConfig:
    """Parametri di §4.2, da `configs/default.yaml` sezione `search`."""

    simulations: int = 400
    c_puct: float = 2.0
    batch_size_leaves: int = 24
    """Quante foglie raccogliere prima di chiamare la rete (§4.3). Senza batching la GPU
    resta ferma il 95% del tempo."""

    virtual_loss: int = 3

    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.0
    """Rumore alla radice. Serve nel self-play della Fase 5 per esplorare aperture
    diverse; giocando sul serio va lasciato a 0, altrimenti si indebolisce di proposito."""

    temperature: float = 0.0
    """0 = gioca sempre la mossa piu visitata. Sopra zero campiona proporzionalmente,
    per la diversita nel self-play."""


@dataclass
class SearchStats:
    """Contatori per il profiling (criticita #6: misurare prima di ottimizzare)."""

    simulations: int = 0
    network_calls: int = 0
    positions_evaluated: int = 0
    terminal_hits: int = 0
    """Foglie risolte senza chiamare la rete perche matto/stallo/patta."""

    collisions: int = 0
    """Volte in cui la selezione e finita su un nodo gia in coda per la valutazione."""

    @property
    def avg_batch(self) -> float:
        return self.positions_evaluated / self.network_calls if self.network_calls else 0.0


@dataclass
class SearchResult:
    move: chess.Move | None
    root: Node
    stats: SearchStats = field(default_factory=SearchStats)

    def visit_counts(self) -> dict[chess.Move, int]:
        return {m: c.visits for m, c in self.root.children.items()}


def policy_target_from(board: chess.Board, result: SearchResult) -> np.ndarray:
    """Distribuzione delle visite come vettore 4672, orientata come in training.

    E il **target della policy** nell'Expert Iteration (Fase 5): la rete imparera a
    imitare cio che la ricerca ha concluso, non cio che pensava prima di cercare.

    Serve la board perche l'indice va orientato dal lato che muove (criticita #1), e il
    nodo da solo non la conserva — tenerla in ogni nodo costerebbe memoria per nulla.
    """
    from chessbot.encoding import N_MOVES, encode_move

    target = np.zeros(N_MOVES, dtype=np.float32)
    total = sum(c.visits for c in result.root.children.values())
    if total == 0:
        return target
    for move, child in result.root.children.items():
        target[encode_move(board, move)] = child.visits / total
    return target


class Evaluator:
    """Valuta posizioni con la rete, in batch.

    Separato dalla ricerca di proposito: cosi si puo sostituire con una rete finta nei
    test (verificando la meccanica dell'albero senza GPU) e con una rete vera in
    produzione, senza toccare l'algoritmo.
    """

    def __init__(self, model: torch.nn.Module, device: torch.device) -> None:
        self.model = model
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def evaluate(self, boards: list[chess.Board]) -> list[tuple[dict[chess.Move, float], float]]:
        """Da posizioni a (priori sulle mosse legali, valore).

        Il valore e in [-1, 1] **dal punto di vista di chi muove**, coerente con la
        convenzione dell'albero.
        """
        if not boards:
            return []

        planes = np.stack([encode_board(b, repetitions=0) for b in boards])
        masks = np.stack([legal_move_mask(b) for b in boards])

        planes_t = torch.from_numpy(planes).to(self.device)
        masks_t = torch.from_numpy(masks).to(self.device)

        policy_logits, wdl_logits = self.model(planes_t)

        # Stessa mascheratura del training (criticita #12).
        masked = masked_policy_logits(policy_logits, masks_t)
        probs = torch.softmax(masked.float(), dim=1).cpu().numpy()

        # Dal WDL al valore scalare: P(vittoria) - P(sconfitta), in [-1, 1].
        wdl = torch.softmax(wdl_logits.float(), dim=1).cpu().numpy()
        values = wdl[:, 2] - wdl[:, 0]

        out: list[tuple[dict[chess.Move, float], float]] = []
        for i, board in enumerate(boards):
            priors: dict[chess.Move, float] = {}
            for move in board.legal_moves:
                from chessbot.encoding import encode_move

                priors[move] = float(probs[i, encode_move(board, move)])
            # Rinormalizza: la softmax e su 4672 indici, ma alcuni indici legali possono
            # collidere in casi limite, e la somma sui soli legali puo non fare 1.
            total = sum(priors.values())
            if total > 0:
                priors = {m: p / total for m, p in priors.items()}
            else:
                uniform = 1.0 / max(len(priors), 1)
                priors = dict.fromkeys(priors, uniform)

            # `values[i]` e dal pov del BIANCO se la board non fosse orientata, ma
            # `encode_board` orienta sempre dal lato che muove: quindi e gia dal pov
            # di chi deve muovere. Nessuna conversione.
            out.append((priors, float(values[i])))
        return out


def expand(node: Node, board: chess.Board, priors: dict[chess.Move, float]) -> None:
    """Crea i figli del nodo, uno per mossa legale."""
    for move, prior in priors.items():
        node.children[move] = Node(prior=prior, to_play=not board.turn)


def add_dirichlet_noise(node: Node, alpha: float, epsilon: float, rng: np.random.Generator) -> None:
    """Rumore di Dirichlet sui priori della radice (§5.2).

    Serve **solo nel self-play**: garantisce che la ricerca provi ogni tanto mosse che la
    policy scarterebbe, cosi le partite generate non sono tutte identiche (criticita #7).
    Giocando sul serio va disattivato.
    """
    if epsilon <= 0 or not node.children:
        return
    moves = list(node.children)
    noise = rng.dirichlet([alpha] * len(moves))
    for move, n in zip(moves, noise, strict=True):
        child = node.children[move]
        child.prior = (1 - epsilon) * child.prior + epsilon * float(n)


def run_mcts(
    board: chess.Board,
    evaluator: Evaluator,
    config: SearchConfig | None = None,
    *,
    rng: np.random.Generator | None = None,
) -> SearchResult:
    """Esegue la ricerca e restituisce la mossa scelta.

    Il ciclo raccoglie piu foglie prima di chiamare la rete (§4.3): con `batch_size_leaves`
    a 24, una chiamata alla GPU serve 24 simulazioni invece di una.
    """
    config = config or SearchConfig()
    rng = rng or np.random.default_rng()
    stats = SearchStats()

    root = Node(prior=1.0, to_play=board.turn)

    # La radice va espansa subito: senza figli non c'e nulla da selezionare.
    priors, _ = evaluator.evaluate([board])[0]
    stats.network_calls += 1
    stats.positions_evaluated += 1
    expand(root, board, priors)
    add_dirichlet_noise(root, config.dirichlet_alpha, config.dirichlet_epsilon, rng)

    if not root.children:
        return SearchResult(None, root, stats)

    while stats.simulations < config.simulations:
        # --- raccogli un batch di foglie -----------------------------------------
        pending: list[tuple[list[Node], chess.Board]] = []
        target = min(config.batch_size_leaves, config.simulations - stats.simulations)

        # I nodi gia in coda per la rete. Senza questo insieme, una foglia in attesa
        # verrebbe riselezionata dalle simulazioni successive — il virtual loss la rende
        # meno attraente ma non la esclude — e il batch si riempirebbe di duplicati
        # invece di esplorare rami nuovi.
        #
        # Misurato prima del fix, su un matto in 1 con 39 mosse legali e 50 simulazioni:
        # con batch 1 la ricerca trovava 43 posizioni terminali e il matto; con batch 8
        # ne trovava ZERO e giocava una mossa a caso. Il batching, che serve a rendere la
        # ricerca piu veloce, la stava rendendo cieca.
        queued: set[int] = set()
        # Cammini che hanno ricevuto virtual loss extra per una collisione: la penalita
        # va tolta a fine batch, altrimenti resta addosso all'albero per sempre e falsa
        # tutte le selezioni successive.
        extra_loss: list[list[Node]] = []
        attempts = 0
        max_attempts = target * 8

        while len(pending) < target and attempts < max_attempts:
            attempts += 1
            path = [root]
            sim_board = board.copy()
            node = root

            # 1. SELEZIONE: scendi finche trovi un nodo non espanso.
            while node.is_expanded:
                move, node = select_child(node, config.c_puct)
                sim_board.push(move)
                path.append(node)

            # Terminale? Risposta esatta, niente rete (§4.3).
            value = terminal_value(sim_board)
            if value is not None:
                backup(path, value)
                stats.simulations += 1
                stats.terminal_hits += 1
                if stats.simulations >= config.simulations:
                    break
                continue

            if id(node) in queued:
                # Gia in coda per la rete. Non si chiude il batch — si aumenta il virtual
                # loss su questo cammino e si riprova: la prossima selezione trovera il
                # ramo meno attraente e andra altrove.
                #
                # > Chiudere il batch qui (il primo tentativo di fix) lo riduceva a UNA
                # > posizione: misurato 199 collisioni su 200 simulazioni, batch medio
                # > 1,0. Il batching esisteva solo sulla carta.
                stats.collisions += 1
                add_virtual_loss(path, config.virtual_loss)
                extra_loss.append(path)
                continue

            # 2. VIRTUAL LOSS: marca il cammino, cosi la prossima selezione va altrove.
            add_virtual_loss(path, config.virtual_loss)
            queued.add(id(node))
            pending.append((path, sim_board))
            stats.simulations += 1

        # Toglie il virtual loss extra accumulato dalle collisioni: quelle simulazioni
        # non hanno prodotto una foglia da valutare, e la penalita non deve sopravvivere
        # al batch.
        for path in extra_loss:
            remove_virtual_loss(path, config.virtual_loss)

        if not pending:
            break

        # --- 3. VALUTAZIONE: una sola chiamata per tutto il batch ------------------
        results = evaluator.evaluate([b for _, b in pending])
        stats.network_calls += 1
        stats.positions_evaluated += len(pending)

        # --- 4. ESPANSIONE e BACKUP ------------------------------------------------
        for (path, sim_board), (priors, value) in zip(pending, results, strict=True):
            remove_virtual_loss(path, config.virtual_loss)
            expand(path[-1], sim_board, priors)
            backup(path, value)

    chosen = select_move(root, config, rng)
    return SearchResult(chosen, root, stats)


def select_move(root: Node, config: SearchConfig, rng: np.random.Generator) -> chess.Move | None:
    """La mossa da giocare: la piu visitata, o campionata se la temperatura e > 0."""
    if not root.children:
        return None

    moves = list(root.children)
    visits = np.array([root.children[m].visits for m in moves], dtype=np.float64)

    if config.temperature <= 0 or visits.sum() == 0:
        return moves[int(visits.argmax())]

    # Temperatura: 1.0 campiona proporzionalmente alle visite, valori bassi si
    # avvicinano all'argmax. Serve nelle prime mosse del self-play (§5.3).
    weights = visits ** (1.0 / config.temperature)
    weights /= weights.sum()
    return moves[int(rng.choice(len(moves), p=weights))]


def principal_variation(root: Node, board: chess.Board, max_depth: int = 10) -> list[chess.Move]:
    """La linea principale: la sequenza di mosse piu visitate.

    Utile per capire cosa sta pensando il motore, e per accorgersi se sta pensando
    qualcosa di assurdo.
    """
    line: list[chess.Move] = []
    node = root
    work = board.copy()
    while node.children and len(line) < max_depth:
        move = max(node.children, key=lambda m: node.children[m].visits)
        if node.children[move].visits == 0:
            break
        line.append(move)
        work.push(move)
        node = node.children[move]
    return line


def describe(result: SearchResult, board: chess.Board, top: int = 5) -> str:
    """Riassunto leggibile della ricerca: le mosse piu visitate e le loro statistiche."""
    children = sorted(result.root.children.items(), key=lambda kv: -kv[1].visits)[:top]
    lines = [
        f"{result.stats.simulations} simulazioni, "
        f"{result.stats.network_calls} chiamate alla rete "
        f"(batch medio {result.stats.avg_batch:.1f}), "
        f"{result.stats.terminal_hits} terminali"
    ]
    for move, child in children:
        pct = 100 * child.visits / max(result.root.visits, 1)
        lines.append(
            f"  {board.san(move):8s} N={child.visits:>5} ({pct:4.1f}%) "
            f"Q={-child.value:+.3f} P={child.prior:.3f}"
        )
    return "\n".join(lines)


__all__ = [
    "Evaluator",
    "SearchConfig",
    "SearchResult",
    "SearchStats",
    "add_dirichlet_noise",
    "describe",
    "expand",
    "policy_target_from",
    "principal_variation",
    "run_mcts",
    "select_move",
]
