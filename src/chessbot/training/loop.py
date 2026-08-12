"""Il ciclo di training supervisionato (§3.3 del piano).

Un ciclo `for` esplicito, senza framework: si legge dall'alto in basso e ogni riga fa
una cosa sola. Le uniche dipendenze sono PyTorch e NumPy.

## Il passo di ottimizzazione, in chiaro

    for batch in loader:
        logit_policy, logit_wdl = rete(batch["planes"])   # forward
        loss = combina(logit_policy, logit_wdl, batch)     # quanto ha sbagliato
        loss.backward()                                    # quanto ogni peso ha colpa
        optimizer.step()                                   # sposta i pesi
        optimizer.zero_grad()                              # azzera per il giro dopo

Tutto il resto — precisione mista, scheduler, checkpoint, logging — e contorno intorno a
queste cinque righe.

## Precisione mista (AMP)

I calcoli girano in fp16 dove si puo, in fp32 dove serve precisione. Sulla 3050 e circa
il doppio piu veloce e dimezza la memoria. Il `GradScaler` esiste perche in fp16 i
gradienti piccoli diventano zero: li moltiplica per un fattore grande prima del backward
e lo toglie dopo, in modo che sopravvivano.

## Learning rate: warmup + cosine

    step 0 -> 2000:  il LR sale da 0 a 1e-3      (warmup)
    poi:             scende da 1e-3 a 0 come un coseno

Il warmup evita che i primi passi, con pesi casuali e gradienti enormi, mandino la rete
fuori strada. Il decay a coseno lascia fare passi grandi all'inizio e piccoli alla fine,
quando si tratta di rifinire.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from chessbot.training.checkpoint import (
    CheckpointMeta,
    HistoryLog,
    TrainingState,
    save_checkpoint,
)
from chessbot.training.loss import LossWeights, accuracy_metrics, compute_loss


@dataclass
class TrainConfig:
    """Iperparametri di §3.3, da `configs/default.yaml` sezione `train`."""

    lr: float = 1.0e-3
    warmup_steps: int = 2000
    weight_decay: float = 1.0e-4
    batch_size: int = 512
    epochs: int = 12
    amp: bool = True
    grad_clip: float = 1.0
    """Norma massima del gradiente. Non e nel piano ma costa nulla: un batch anomalo puo
    produrre un gradiente enorme che sposta i pesi in un punto da cui non si torna."""

    log_every_steps: int = 50
    eval_every_steps: int = 2000
    checkpoint_every_steps: int = 2000
    max_steps: int | None = None
    """Fermata anticipata, per gli smoke test. None = fino alla fine delle epoche."""


def build_optimizer(model: nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    """AdamW, con weight decay solo dove ha senso.

    Il decay tira i pesi verso zero, il che regolarizza le convoluzioni e i layer lineari.
    Applicarlo ai bias e ai parametri di BatchNorm invece fa danni: quelli devono poter
    assumere il valore che serve, e spingerli a zero e una preferenza arbitraria. E una
    correzione standard, spesso dimenticata perche AdamW di default non la fa.
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)

    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=config.lr,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer, config: TrainConfig, total_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    """Warmup lineare seguito da decay a coseno."""
    warmup = max(1, config.warmup_steps)

    def factor(step: int) -> float:
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        progress = min(1.0, progress)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    """Sposta un batch sul device. `non_blocking` funziona solo con pin_memory."""
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    weights: LossWeights,
    *,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Valuta su un loader, restituendo loss e metriche medie.

    `model.eval()` non e cosmetico: BatchNorm in training usa le statistiche del batch
    corrente, in eval quelle accumulate. Dimenticarlo fa dipendere la valutazione dalla
    composizione casuale dei batch, e i numeri ballano senza motivo apparente.
    """
    model.eval()
    totals = {"loss": 0.0, "policy": 0.0, "wdl": 0.0, "top1": 0.0, "top5": 0.0, "wdl_acc": 0.0}
    n = 0

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = move_batch(batch, device)
        policy_logits, wdl_logits = model(batch["planes"])
        parts = compute_loss(policy_logits, wdl_logits, batch, weights)
        metrics = accuracy_metrics(policy_logits, wdl_logits, batch)

        totals["loss"] += parts.total.item()
        totals["policy"] += parts.policy.item()
        totals["wdl"] += parts.wdl.item()
        for k, v in metrics.items():
            totals[k] += v
        n += 1

    model.train()
    return {k: v / max(n, 1) for k, v in totals.items()}


def train(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    *,
    config: TrainConfig,
    device: torch.device,
    val_loader: torch.utils.data.DataLoader | None = None,
    weights: LossWeights | None = None,
    run_dir: Path | None = None,
    state: TrainingState | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
    on_log: Callable[[dict], None] | None = None,
) -> TrainingState:
    """Esegue il training. Restituisce lo stato finale.

    Gli oggetti optimizer/scheduler/scaler si possono passare dall'esterno: e cosi che
    `scripts/train.py` riprende un run interrotto, ricaricandoli dal checkpoint prima di
    chiamare questa funzione.
    """
    weights = weights or LossWeights()
    state = state or TrainingState()

    steps_per_epoch = len(train_loader)
    total_steps = config.max_steps or steps_per_epoch * config.epochs

    optimizer = optimizer or build_optimizer(model, config)
    scheduler = scheduler or build_scheduler(optimizer, config, total_steps)
    use_amp = config.amp and device.type == "cuda"
    scaler = scaler or torch.amp.GradScaler(device.type, enabled=use_amp)

    history = HistoryLog(run_dir / "history.jsonl") if run_dir else None

    model.to(device)
    model.train()

    started = time.perf_counter() - state.elapsed_seconds
    last_log = 0.0
    running: dict[str, float] = {}
    running_n = 0

    print(f"  {steps_per_epoch:,} step per epoca, {total_steps:,} totali")
    print(f"  device {device}, precisione mista {'attiva' if use_amp else 'disattiva'}")
    print()

    stop = False
    for epoch in range(state.epoch, config.epochs):
        if stop:
            break
        state.epoch = epoch

        for batch in train_loader:
            batch = move_batch(batch, device)

            # --- i cinque passi ---------------------------------------------------
            with torch.autocast(device.type, enabled=use_amp):
                policy_logits, wdl_logits = model(batch["planes"])
                parts = compute_loss(policy_logits, wdl_logits, batch, weights)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(parts.total).backward()

            # Il clipping va fatto sui gradienti NON scalati: unscale_ li riporta alla
            # scala vera prima di misurarne la norma.
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            # ----------------------------------------------------------------------

            state.global_step += 1
            state.samples_seen += batch["planes"].shape[0]
            state.elapsed_seconds = time.perf_counter() - started

            for k, v in (
                ("loss", parts.total.item()),
                ("policy", parts.policy.item()),
                ("wdl", parts.wdl.item()),
                ("eval_mse", parts.eval_mse.item()),
            ):
                running[k] = running.get(k, 0.0) + v
            running_n += 1

            if state.global_step % config.log_every_steps == 0:
                now = time.perf_counter()
                rate = (
                    config.log_every_steps * config.batch_size / (now - last_log) if last_log else 0
                )
                avg = {k: v / running_n for k, v in running.items()}
                record = {
                    "kind": "train",
                    "step": state.global_step,
                    "epoch": epoch,
                    "lr": scheduler.get_last_lr()[0],
                    "samples_per_s": round(rate),
                    **{k: round(v, 4) for k, v in avg.items()},
                }
                print(
                    f"  step {state.global_step:>7,} | ep {epoch} | "
                    f"loss {avg['loss']:.4f} (pol {avg['policy']:.4f}, "
                    f"wdl {avg['wdl']:.4f}) | lr {record['lr']:.2e} | "
                    f"{rate:.0f} pos/s",
                    flush=True,
                )
                if history:
                    history.append(record)
                if on_log:
                    on_log(record)
                running, running_n = {}, 0
                last_log = now

            if val_loader is not None and state.global_step % config.eval_every_steps == 0:
                metrics = evaluate(model, val_loader, device, weights, max_batches=40)
                print(
                    f"  {'':>7}   VAL   | loss {metrics['loss']:.4f} | "
                    f"top-1 {100 * metrics['top1']:.2f}% | top-5 {100 * metrics['top5']:.2f}% | "
                    f"wdl {100 * metrics['wdl_acc']:.1f}%",
                    flush=True,
                )
                if history:
                    history.append(
                        {"kind": "val", "step": state.global_step, "epoch": epoch, **metrics}
                    )

                if run_dir and metrics["top1"] > state.best_val_top1:
                    state.best_val_top1 = metrics["top1"]
                    save_checkpoint(
                        run_dir / "best.pt",
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        state=state,
                        meta=CheckpointMeta(notes=f"best top-1 {100 * metrics['top1']:.2f}%"),
                    )

            if run_dir and state.global_step % config.checkpoint_every_steps == 0:
                save_checkpoint(
                    run_dir / "last.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    state=state,
                )

            if config.max_steps and state.global_step >= config.max_steps:
                stop = True
                break

    # Salvataggio finale: senza, gli ultimi step fra un checkpoint e l'altro si perdono.
    if run_dir:
        save_checkpoint(
            run_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            state=state,
            meta=CheckpointMeta(notes="fine del training"),
        )

    return state


__all__ = [
    "TrainConfig",
    "build_optimizer",
    "build_scheduler",
    "evaluate",
    "move_batch",
    "train",
]
