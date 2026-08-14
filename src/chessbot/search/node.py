"""Il nodo dell'albero MCTS e la formula PUCT (§4.2 del piano).

Un nodo e una posizione. I suoi figli sono le mosse legali da li. Ogni ramo tiene tre
numeri, e sono tutto cio che serve:

| | |
|---|---|
| `N` | quante volte quella mossa e stata visitata |
| `W` | somma dei valori osservati scendendo di li |
| `P` | quanto la rete la reputa buona a colpo d'occhio (prior) |

`Q = W / N` e il valore medio. Con `N = 0` non c'e ancora nulla da mediare.

## La convenzione del segno — criticita #2

> **Il valore e SEMPRE dal punto di vista di chi deve muovere nel nodo.**

Scendendo l'albero il giocatore si alterna, quindi risalendo il valore va **negato ad
ogni livello**:

    foglia:  +0.7   bene per chi muove LI
    padre:   -0.7   male per chi muove li, e l'avversario
    nonno:   +0.7

Sbagliare questo segno produce un motore che **gioca contro se stesso**. Non crasha, non
lancia eccezioni: sembra solo "un po' debole". E il bug piu insidioso del progetto, e il
test di monotonicita del Gate 5 esiste apposta per intercettarlo.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import chess

# Valore di una posizione terminale, dal punto di vista di chi deve muovere.
# Chi e sotto matto ha perso: -1. Patta: 0.
TERMINAL_LOSS = -1.0
TERMINAL_DRAW = 0.0


@dataclass(slots=True)
class Node:
    """Una posizione nell'albero di ricerca.

    I figli si creano solo quando il nodo viene espanso: un albero con 400 simulazioni
    ha al piu 400 nodi, non 30^400.
    """

    prior: float
    """P(s,a): probabilita che la policy della rete assegna alla mossa che porta qui."""

    to_play: chess.Color
    """Chi deve muovere in questa posizione. Serve per interpretare il segno di Q."""

    visits: int = 0
    value_sum: float = 0.0
    children: dict[chess.Move, Node] = field(default_factory=dict)

    virtual_loss: int = 0
    """Penalita temporanea durante il batching (§4.3): quando piu simulazioni scendono
    in parallelo, questa impedisce che finiscano tutte sullo stesso ramo."""

    @property
    def is_expanded(self) -> bool:
        return bool(self.children)

    @property
    def value(self) -> float:
        """Q(s,a) = W / N, dal punto di vista di chi muove QUI.

        Con zero visite non c'e nulla da mediare: si restituisce 0, che in PUCT
        equivale a "sconosciuto, decide il prior".

        > **Il segno del virtual loss.** Il valore di questo nodo e dal punto di vista di
        > chi muove QUI, ma chi lo *sceglie* e il genitore, cioe l'avversario: il PUCT
        > infatti usa `-child.value`.
        >
        > La penalita va quindi applicata nel verso che il genitore percepisce come
        > peggioramento — cioe **sommando**, non sottraendo. Sottraendo (il primo
        > tentativo) il virtual loss diventava un premio: misurato, un nodo penalizzato
        > quattro volte passava da PUCT 0,1 a 1,1 e veniva riselezionato all'infinito.
        > Risultato: 24.433 collisioni su 200 simulazioni e batch medio 1,0.
        """
        effective = self.visits + self.virtual_loss
        if effective == 0:
            return 0.0
        return (self.value_sum + self.virtual_loss) / effective


def puct_score(child: Node, parent_visits: int, c_puct: float) -> float:
    """Il punteggio PUCT di un figlio (§4.2).

        Q(s,a) + c_puct * P(s,a) * sqrt(N_padre) / (1 + N(s,a))

    Due spinte opposte:

    - **Q** premia cio che finora ha funzionato — sfruttamento
    - il secondo termine premia cio che e poco visitato ma la rete reputa buono —
      esplorazione

    `N` sta al denominatore: piu visiti un ramo, meno vale esplorarlo ancora. Cosi la
    ricerca non si incaponisce su una linea sola ne si disperde su tutte.

    > **Il segno di Q.** `child.value` e dal punto di vista di chi muove NEL FIGLIO,
    > cioe l'avversario di chi sta scegliendo. Va quindi negato: cio che e buono per lui
    > e cattivo per me.
    """
    exploration = c_puct * child.prior * math.sqrt(parent_visits) / (1 + child.visits)
    return -child.value + exploration


def select_child(node: Node, c_puct: float) -> tuple[chess.Move, Node]:
    """Il figlio col punteggio PUCT piu alto."""
    parent_visits = max(1, node.visits + node.virtual_loss)
    return max(
        node.children.items(),
        key=lambda item: puct_score(item[1], parent_visits, c_puct),
    )


def terminal_value(board: chess.Board) -> float | None:
    """Valore di una posizione terminale, o None se la partita continua.

    > §4.3 — Riconoscere matto, stallo, ripetizione e regola delle 50 mosse **prima** di
    > chiamare la rete. Sono risposte esatte che costano nulla, mentre un forward pass
    > costa millisecondi e darebbe una stima approssimata di qualcosa di gia certo.
    """
    if board.is_checkmate():
        # Chi deve muovere e sotto matto: ha perso.
        return TERMINAL_LOSS
    if (
        board.is_stalemate()
        or board.is_insufficient_material()
        or board.is_repetition(3)
        or board.can_claim_fifty_moves()
    ):
        return TERMINAL_DRAW
    return None


def backup(path: list[Node], value: float) -> None:
    """Risale il cammino aggiornando le statistiche, negando il valore ad ogni livello.

    > **Criticita #2 in tre righe.** `value` arriva dal punto di vista di chi muove nella
    > FOGLIA. Il padre della foglia e l'avversario, quindi per lui vale l'opposto; il
    > nonno di nuovo l'opposto, e cosi via.
    >
    > Il ciclo va quindi percorso dalla foglia alla radice negando ogni volta. Percorrerlo
    > senza negare, o negando una volta sola, produce un motore che valuta le proprie
    > posizioni con il segno dell'avversario.
    """
    for node in reversed(path):
        node.visits += 1
        node.value_sum += value
        value = -value


def add_virtual_loss(path: list[Node], amount: int) -> None:
    """Marca un cammino come "in corso di valutazione" (§4.3).

    Durante il batching piu simulazioni scendono nell'albero prima che arrivi una sola
    risposta dalla rete. Senza questa penalita finirebbero tutte sullo stesso ramo — il
    migliore secondo le statistiche attuali — e il batch conterrebbe N copie della stessa
    posizione, rendendo il batching inutile.
    """
    for node in path:
        node.virtual_loss += amount


def remove_virtual_loss(path: list[Node], amount: int) -> None:
    """Toglie la penalita, quando la valutazione vera e arrivata."""
    for node in path:
        node.virtual_loss -= amount


__all__ = [
    "TERMINAL_DRAW",
    "TERMINAL_LOSS",
    "Node",
    "add_virtual_loss",
    "backup",
    "puct_score",
    "remove_virtual_loss",
    "select_child",
    "terminal_value",
]
