"""Expert Iteration: il ciclo dell'RL (§5.1 del piano).

    per ogni iterazione:
        1. SELF-PLAY   la rete gioca contro se stessa con MCTS
        2. TRAINING    si riallena sulle distribuzioni di visite prodotte
        3. GATING      la rete nuova sfida la vecchia: se vince, la sostituisce

**La loss e identica a quella della Fase 3.** Cambia solo la provenienza delle
etichette: prima gli umani di Lichess, ora l'MCTS di se stessi. Non serve un algoritmo
nuovo — tutta la complessita sta nell'infrastruttura di generazione partite.

## Perche funziona

L'MCTS e un **operatore di miglioramento della policy garantito**: la distribuzione delle
visite π e dimostrabilmente migliore della policy p, perche e p **piu** N simulazioni di
verifica. La rete impara a produrre subito cio che prima le costava una ricerca, e la
ricerca successiva riparte da una policy migliore.

Non serve stimare la direzione del miglioramento come farebbe PPO: la ricerca la
fornisce. E un vantaggio disponibile solo con un simulatore perfetto, che negli scacchi
c'e.

## Le due differenze rispetto ad AlphaZero originale

- **penalita KL** verso la policy supervisionata: loro partivano da zero e non avevano
  nulla da dimenticare, qui si parte da una rete che sa gia le aperture umane e non deve
  buttarle via inseguendo strategie degeneri di self-play
- **target del valore misto** (§5.5): `alpha*Q_radice + (1-alpha)*z` invece del solo
  risultato, perche Q e una stima cercata e molto meno rumorosa
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import chess
import numpy as np
import torch
import torch.nn.functional as F

from chessbot.encoding import N_MOVES, encode_board, encode_move
from chessbot.model.network import masked_policy_logits
from chessbot.search.parallel import GameRecord, value_target


@dataclass
class RLConfig:
    """Parametri di §5.2, da `configs/rl.yaml`."""

    games_per_iteration: int = 500
    simulations_per_move: int = 175
    parallel_games: int = 32
    """Misurato: satura a 32 su questa macchina, oltre il batch cresce ma la velocita no."""

    dirichlet_alpha: float = 0.3
    dirichlet_weight: float = 0.25
    temperature_moves: int = 15

    replay_buffer_iterations: int = 9
    epochs_per_iteration: int = 3

    lr: float = 1.0e-4
    momentum: float = 0.9
    batch_size: int = 256

    kl_penalty_weight: float = 0.075
    """Ancora la rete alla policy supervisionata: senza, il self-play puo derivare verso
    strategie degeneri che funzionano solo contro se stesse."""

    alpha_q_root: float = 0.4
    max_draw_rate: float = 0.70
    max_plies: int = 300

    # Criteri di stop (criticita #9), fissati PRIMA di iniziare.
    max_iterations: int = 40
    abort_if_no_gating_pass_within: int = 10
    abort_if_cumulative_elo_below: float = 50.0
    abort_elo_check_at: int = 25


@dataclass
class ReplayBuffer:
    """Le posizioni delle ultime N iterazioni.

    Non solo l'ultima: allenarsi solo sui dati piu freschi rende il training instabile,
    perche la distribuzione cambia ad ogni iterazione. Tenere una finestra di 9
    iterazioni smorza le oscillazioni.
    """

    max_iterations: int = 9
    chunks: list[list[GameRecord]] = field(default_factory=list)

    def add(self, records: list[GameRecord]) -> None:
        self.chunks.append(records)
        while len(self.chunks) > self.max_iterations:
            self.chunks.pop(0)

    def all_records(self) -> list[GameRecord]:
        return [r for chunk in self.chunks for r in chunk]

    def __len__(self) -> int:
        return sum(len(c) for c in self.chunks)


def records_to_tensors(
    records: list[GameRecord], config: RLConfig
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Da record di self-play a tensori per il training.

    Returns:
        `(planes, policy_target, value_target, legal_mask)`.

    Il target della policy e la distribuzione delle visite normalizzata, **non** un
    indice: la rete impara l'intera distribuzione, che porta molta piu informazione della
    sola mossa scelta. E la differenza fra "gioca e4" e "e4 al 60%, d4 al 25%, il resto
    trascurabile".
    """
    n = len(records)
    planes = np.zeros((n, 19, 8, 8), dtype=np.float32)
    policy = np.zeros((n, N_MOVES), dtype=np.float32)
    values = np.zeros(n, dtype=np.float32)
    masks = np.zeros((n, N_MOVES), dtype=bool)

    for i, record in enumerate(records):
        board = chess.Board(record.fen)
        planes[i] = encode_board(board, repetitions=0)

        total = sum(record.visit_counts.values())
        for move, visits in record.visit_counts.items():
            idx = encode_move(board, move)
            masks[i, idx] = True
            if total > 0:
                policy[i, idx] = visits / total

        values[i] = value_target(record, config.alpha_q_root)

    return (
        torch.from_numpy(planes),
        torch.from_numpy(policy),
        torch.from_numpy(values),
        torch.from_numpy(masks),
    )


def rl_loss(
    policy_logits: torch.Tensor,
    wdl_logits: torch.Tensor,
    policy_target: torch.Tensor,
    value_target_t: torch.Tensor,
    legal_mask: torch.Tensor,
    reference_logits: torch.Tensor | None,
    config: RLConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """La loss dell'Expert Iteration.

        L = CE(policy, π) + MSE(valore, target) + kl * KL(policy || policy_supervisionata)

    Il primo termine e una cross-entropy con target **distribuzione**, non indice: si
    calcola come `-Σ π log p`. Gli altri due sono l'ancoraggio al valore cercato e alla
    policy di partenza.
    """
    masked = masked_policy_logits(policy_logits, legal_mask)
    log_probs = torch.log_softmax(masked, dim=1)

    # Cross-entropy fra due distribuzioni: -sum(target * log_pred).
    policy_loss = -(policy_target * log_probs).sum(dim=1).mean()

    # Il valore scalare atteso dalla testa WDL: P(vittoria) - P(sconfitta).
    wdl_probs = torch.softmax(wdl_logits, dim=1)
    scalar_value = wdl_probs[:, 2] - wdl_probs[:, 0]
    value_loss = F.mse_loss(scalar_value, value_target_t)

    total = policy_loss + value_loss
    kl_value = 0.0

    if reference_logits is not None and config.kl_penalty_weight > 0:
        # > **La penalita KL non e in AlphaZero.** Loro partivano da pesi casuali e non
        # > avevano nulla da dimenticare. Qui si parte da una rete che sa le aperture
        # > umane, e il self-play puo derivare verso strategie che funzionano solo contro
        # > se stesse. La KL tiene la policy nuova vicina a quella supervisionata.
        ref_masked = masked_policy_logits(reference_logits, legal_mask)
        ref_log_probs = torch.log_softmax(ref_masked, dim=1)
        ref_probs = ref_log_probs.exp()
        kl = (ref_probs * (ref_log_probs - log_probs)).sum(dim=1).mean()
        total = total + config.kl_penalty_weight * kl
        kl_value = float(kl.detach())

    return total, {
        "policy": float(policy_loss.detach()),
        "value": float(value_loss.detach()),
        "kl": kl_value,
        "total": float(total.detach()),
    }


def build_rl_optimizer(model: torch.nn.Module, config: RLConfig) -> torch.optim.Optimizer:
    """SGD con momentum, non Adam.

    > §5.2 lo specifica: **piu stabile di Adam sotto distribution shift**. I dati di
    > self-play cambiano distribuzione ad ogni iterazione, e le medie mobili adattive di
    > Adam inseguono quel cambiamento amplificandolo. SGD e piu lento ma non oscilla.
    """
    return torch.optim.SGD(
        model.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=1e-4
    )


@dataclass
class IterationResult:
    iteration: int
    games: int
    positions: int
    draw_rate: float
    gating_accepted: bool
    gating_summary: str
    elo_estimate: float
    losses: dict[str, float]
    elapsed_seconds: float


def should_stop(results: list[IterationResult], config: RLConfig) -> tuple[bool, str]:
    """I criteri di stop di §5.6, fissati prima di iniziare (criticita #9).

    > Un ciclo RL senza condizione di abbandono definita in anticipo diventa un pozzo di
    > tempo.
    """
    n = len(results)
    if n == 0:
        return False, ""

    if n >= config.max_iterations:
        return True, f"tetto duro: {config.max_iterations} iterazioni raggiunte"

    promossi = [r for r in results if r.gating_accepted]
    if n >= config.abort_if_no_gating_pass_within and not promossi:
        return True, (
            f"gating mai passato in {n} iterazioni "
            f"(limite {config.abort_if_no_gating_pass_within}): fermarsi e analizzare"
        )

    if n >= config.abort_elo_check_at:
        cumulativo = sum(r.elo_estimate for r in results if r.gating_accepted)
        if cumulativo < config.abort_if_cumulative_elo_below:
            return True, (
                f"guadagno cumulativo {cumulativo:+.0f} Elo dopo {n} iterazioni, "
                f"sotto la soglia di {config.abort_if_cumulative_elo_below:.0f}"
            )

    return False, ""


def load_rl_config(path: Path) -> RLConfig:
    """Legge `configs/rl.yaml` nella dataclass."""
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    rl = data.get("rl", {})

    return RLConfig(
        games_per_iteration=rl.get("games_per_iteration", 500),
        simulations_per_move=rl.get("simulations_per_move", 175),
        parallel_games=rl.get("parallel_games", 32),
        dirichlet_alpha=rl.get("dirichlet_alpha", 0.3),
        dirichlet_weight=rl.get("dirichlet_weight", 0.25),
        temperature_moves=rl.get("temperature_moves", 15),
        replay_buffer_iterations=rl.get("replay_buffer_iterations", 9),
        epochs_per_iteration=rl.get("epochs_per_iteration", 3),
        lr=rl.get("lr", 1.0e-4),
        momentum=rl.get("momentum", 0.9),
        kl_penalty_weight=rl.get("kl_penalty_weight", 0.075),
        alpha_q_root=rl.get("value_target", {}).get("alpha_q_root", 0.4),
        max_draw_rate=rl.get("openings", {}).get("max_draw_rate", 0.70),
        max_iterations=rl.get("stop_criteria", {}).get("max_iterations", 40),
        abort_if_no_gating_pass_within=rl.get("stop_criteria", {}).get(
            "abort_if_no_gating_pass_within", 10
        ),
        abort_if_cumulative_elo_below=rl.get("stop_criteria", {}).get(
            "abort_if_cumulative_elo_below", 50
        ),
        abort_elo_check_at=rl.get("stop_criteria", {}).get("abort_elo_check_at", 25),
    )


__all__ = [
    "IterationResult",
    "RLConfig",
    "ReplayBuffer",
    "build_rl_optimizer",
    "load_rl_config",
    "records_to_tensors",
    "rl_loss",
    "should_stop",
]
