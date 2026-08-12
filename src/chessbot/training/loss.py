"""Loss composita e metriche del training supervisionato (§3.2 del piano).

    L = CE(policy, mossa_giocata)
      + 0.5 * CE(wdl, esito_partita)
      + 0.5 * MSE(value, eval_stockfish)     # solo dove disponibile

Tre segnali diversi sulla stessa posizione:

- **policy**: quale mossa ha giocato un umano forte. E il segnale principale, quello che
  la rete usera per proporre mosse.
- **WDL**: come e finita la partita. Segnale rumoroso — una posizione vinta si puo
  perdere venti mosse dopo — ma abbondante, c'e su tutte le posizioni.
- **eval Stockfish**: quanto vale davvero la posizione, secondo un motore forte. Molto
  meno rumoroso del WDL, ma disponibile solo sul 33,2% dei campioni.

I pesi 0.5 vengono da `configs/default.yaml`. La policy pesa il doppio perche e cio che
il motore usera per giocare; le altre due teste servono all'MCTS (Stadio 5), dove il
valore delle foglie decide la ricerca.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from chessbot.model.network import masked_policy_logits


@dataclass
class LossWeights:
    """Pesi di §3.2, da `configs/default.yaml` sezione `train.loss_weights`."""

    policy: float = 1.0
    wdl: float = 0.5
    eval_mse: float = 0.5
    label_smoothing: float = 0.05


@dataclass
class LossParts:
    """I termini separati, per poterli guardare uno per uno su TensorBoard.

    Tenerli distinti non e cosmetico: se la loss totale non scende, sapere QUALE dei tre
    termini e fermo dice subito dove guardare.
    """

    total: torch.Tensor
    policy: torch.Tensor
    wdl: torch.Tensor
    eval_mse: torch.Tensor
    eval_coverage: float
    """Frazione del batch che aveva una valutazione Stockfish. Sotto il 30% circa il
    termine eval_mse e stimato su pochi campioni e va letto con cautela."""


def masked_cross_entropy(
    masked_logits: torch.Tensor,
    target: torch.Tensor,
    legal_mask: torch.Tensor,
    *,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Cross-entropy con label smoothing distribuito **solo sulle mosse legali**.

    > **Perche non si puo usare `F.cross_entropy(..., label_smoothing=...)`.**
    > Quella funzione spalma la massa di smoothing su TUTTE le classi, comprese le
    > migliaia di mosse illegali che la mascheratura ha portato a -1e9. Con 4672 classi
    > di cui ~30 legali, il conto e:
    >
    >     0.05 / 4672 * 4642 classi illegali * 1e9 = 49.678.938
    >
    > Misurato: la loss valeva **49.683.864** invece di ~3,4. Il gradiente sui logit
    > legali resta corretto — l'overfit test passava lo stesso al 100% — ma la loss e
    > illeggibile, il termine WDL viene schiacciato di sette ordini di grandezza, e ogni
    > soglia numerica del Gate 4 diventa priva di senso.
    >
    > Il bug e silenzioso nel modo peggiore: non impedisce alla rete di imparare, rende
    > solo impossibile *accorgersi* se sta imparando.

    Qui lo smoothing si applica alla sola distribuzione legale: il target diventa
    `1 - eps` sulla mossa giocata ed `eps / (n_legali - 1)` sulle altre mosse legali.
    """
    log_probs = torch.log_softmax(masked_logits, dim=1)
    nll = -log_probs.gather(1, target.unsqueeze(1)).squeeze(1)

    if label_smoothing <= 0.0:
        return nll.mean()

    n_legal = legal_mask.sum(dim=1).clamp(min=1)

    # Media dei log-probs sulle sole caselle legali. I logit illegali valgono -1e9,
    # quindi i loro log_probs sono ~-1e9: vanno esclusi dalla somma, non pesati.
    legal_log_probs = torch.where(legal_mask, log_probs, torch.zeros_like(log_probs))
    mean_legal = legal_log_probs.sum(dim=1) / n_legal

    # Con una sola mossa legale non c'e nulla su cui distribuire: lo smoothing si annulla
    # da solo perche mean_legal coincide con nll.
    smooth = -mean_legal
    return ((1.0 - label_smoothing) * nll + label_smoothing * smooth).mean()


def compute_loss(
    policy_logits: torch.Tensor,
    wdl_logits: torch.Tensor,
    batch: dict[str, torch.Tensor],
    weights: LossWeights | None = None,
) -> LossParts:
    """Calcola i tre termini e la loro somma pesata.

    > **Criticita #12 — la mascheratura delle mosse illegali avviene QUI**, con la stessa
    > funzione usata in inferenza (`masked_policy_logits`). Senza, la cross-entropy
    > spingerebbe la rete a distribuire probabilita anche su mosse impossibili, e in
    > inferenza si troverebbe a normalizzare su un insieme diverso da quello su cui e
    > stata allenata.
    """
    weights = weights or LossWeights()

    # --- policy ---------------------------------------------------------------------
    masked = masked_policy_logits(policy_logits, batch["legal_mask"])
    policy_loss = masked_cross_entropy(
        masked,
        batch["move_index"],
        batch["legal_mask"],
        label_smoothing=weights.label_smoothing,
    )

    # --- WDL ------------------------------------------------------------------------
    # Niente label smoothing qui: l'esito di una partita e un fatto, non una preferenza
    # fra alternative plausibili come lo e la scelta di una mossa.
    wdl_loss = F.cross_entropy(wdl_logits, batch["wdl"])

    # --- eval Stockfish, solo dove c'e ------------------------------------------------
    eval_target = batch["eval_cp"]
    has_eval = ~torch.isnan(eval_target)
    coverage = has_eval.float().mean().item()

    if has_eval.any():
        # Dalla distribuzione WDL si ricava un valore scalare atteso in [-1, 1]:
        #   P(vittoria) * (+1) + P(patta) * 0 + P(sconfitta) * (-1)
        # E la stessa quantita che l'eval di Stockfish esprime, quindi sono confrontabili.
        wdl_probs = torch.softmax(wdl_logits, dim=1)
        scalar_value = wdl_probs[:, 2] - wdl_probs[:, 0]
        eval_loss = F.mse_loss(
            scalar_value[has_eval],
            torch.tanh(eval_target[has_eval]),
        )
    else:
        # Nessun campione con eval in questo batch: contributo nullo, ma il tensore deve
        # esistere e stare sullo stesso device per la somma.
        eval_loss = torch.zeros((), device=policy_logits.device)

    total = weights.policy * policy_loss + weights.wdl * wdl_loss + weights.eval_mse * eval_loss

    return LossParts(
        total=total,
        policy=policy_loss.detach(),
        wdl=wdl_loss.detach(),
        eval_mse=eval_loss.detach(),
        eval_coverage=coverage,
    )


@torch.no_grad()
def accuracy_metrics(
    policy_logits: torch.Tensor,
    wdl_logits: torch.Tensor,
    batch: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Metriche leggibili: top-1 e top-5 sulla policy, accuracy sul WDL.

    La top-1 e la metrica del Gate 4: deve arrivare fra il 45% e il 52%. Sotto il 45% il
    piano dice di sospettare la criticita #1 (etichette specchiate) — nel nostro caso
    l'abbiamo gia esclusa al 100,0000% su 20M campioni, quindi un valore basso
    indicherebbe un problema altrove: loss, learning rate o mascheratura.

    Un riferimento per leggere il numero: predire sempre la mossa piu frequente da circa
    il 3-4%, e una scelta a caso fra le legali circa il 3%. Il 50% e molto piu alto di
    quanto sembri.
    """
    masked = masked_policy_logits(policy_logits, batch["legal_mask"])
    target = batch["move_index"]

    top1 = (masked.argmax(dim=1) == target).float().mean().item()

    k = min(5, masked.shape[1])
    top5_indices = masked.topk(k, dim=1).indices
    top5 = (top5_indices == target.unsqueeze(1)).any(dim=1).float().mean().item()

    wdl_acc = (wdl_logits.argmax(dim=1) == batch["wdl"]).float().mean().item()

    return {"top1": top1, "top5": top5, "wdl_acc": wdl_acc}


__all__ = [
    "LossParts",
    "LossWeights",
    "accuracy_metrics",
    "compute_loss",
    "masked_cross_entropy",
]
