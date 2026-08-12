"""Salvataggio e ripresa del training (§Fase 1 del piano, "Infrastruttura da predisporre").

> Il checkpointing non e opzionale: su un portatile un aggiornamento di sistema o uno
> shutdown termico azzerano una notte di training.

## Cosa contiene un checkpoint

Tutto quello che serve per **riprendere esattamente da dove si era interrotto**, non solo
per rileggere i pesi:

| Campo | Perche serve |
|---|---|
| pesi del modello | ovvio |
| stato dell'optimizer | AdamW tiene medie mobili dei gradienti: senza, riparti con momenti azzerati e la loss ha un salto |
| stato dello scheduler | il learning rate segue una curva: ripartire dal LR iniziale rovina il decay |
| stato del GradScaler | la precisione mista ha un fattore di scala adattivo, anche lui va ripreso |
| epoca e step globale | per sapere a che punto della curva si e |
| stato RNG (python, numpy, torch, cuda) | perche l'ordine dei dati e le augmentation riprendano identici |
| config e commit hash | regola #2: un risultato che non sai riprodurre non e un risultato |

## Scrittura atomica

Si scrive su un file temporaneo e poi lo si rinomina. `os.replace` e atomico: o il file
c'e intero, o resta quello vecchio. Senza questa accortezza, un crash **durante** il
salvataggio lascerebbe un checkpoint troncato — e lo scopriresti al riavvio, avendo perso
sia il vecchio sia il nuovo.

## Quali file restano su disco

    runs/<nome>/
      last.pt          l'ultimo, sovrascritto ad ogni salvataggio
      best.pt          quello con la top-1 di validation piu alta
      step_002000.pt   snapshot periodici, se `keep_every` e attivo
      history.jsonl    una riga per valutazione: append-only, non si perde mai

`last.pt` serve a riprendere, `best.pt` a giocare. Sono file diversi di proposito:
l'ultimo checkpoint non e necessariamente il migliore.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]

# Versione dello schema del checkpoint. Se un domani i campi cambiano, questo permette di
# accorgersene al caricamento invece di leggere spazzatura.
SCHEMA_VERSION = 1


def commit_hash() -> str:
    """Hash del commit corrente, per legare i pesi al codice che li ha prodotti."""
    try:
        p = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        h = p.stdout.strip() or "sconosciuto"
        # Un checkpoint prodotto con modifiche non committate non e riproducibile: va
        # marcato, altrimenti fra un mese il commit hash sarebbe una bugia.
        return f"{h}-dirty" if dirty.stdout.strip() else h
    except Exception:
        return "sconosciuto"


@dataclass
class TrainingState:
    """Il punto esatto in cui si trova il training."""

    epoch: int = 0
    global_step: int = 0
    samples_seen: int = 0
    best_val_top1: float = 0.0
    elapsed_seconds: float = 0.0


@dataclass
class CheckpointMeta:
    """Contorno informativo: serve a capire cos'e questo file fra sei mesi."""

    schema_version: int = SCHEMA_VERSION
    commit: str = field(default_factory=commit_hash)
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    config: dict[str, Any] = field(default_factory=dict)
    notes: str = ""


def rng_state() -> dict[str, Any]:
    """Fotografa lo stato di tutti i generatori casuali."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Rimette i generatori nello stato salvato.

    > **Lo stato RNG deve stare su CPU come ByteTensor.** Caricando un checkpoint con
    > `map_location="cuda"`, PyTorch sposta *tutti* i tensori sulla GPU — stato RNG
    > compreso — e `set_rng_state` fallisce con
    > `TypeError: RNG state must be a torch.ByteTensor`. Si riportano quindi
    > esplicitamente su CPU prima di usarli.
    """
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])

    torch.set_rng_state(state["torch"].cpu().to(torch.uint8))
    if "cuda" in state and torch.cuda.is_available():
        cuda_states = [s.cpu().to(torch.uint8) for s in state["cuda"]]
        # Se il checkpoint viene da una macchina con piu GPU, si ripristina solo cio che
        # esiste qui: meglio un ripristino parziale che un crash.
        if len(cuda_states) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(cuda_states)


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    scaler: Any = None,
    state: TrainingState | None = None,
    meta: CheckpointMeta | None = None,
) -> Path:
    """Salva in modo atomico. Restituisce il percorso scritto.

    Il modello viene salvato con `state_dict()`, non con `torch.save(model)`: il secondo
    serializza anche la classe, e diventa illeggibile se un domani il codice cambia nome
    o si sposta di modulo.
    """
    # La config dell'architettura viaggia col checkpoint: senza, ricaricare i pesi
    # richiederebbe di ricordarsi a mano quanti blocchi e canali aveva la rete.
    config = getattr(model, "config", None)
    model_config = asdict(config) if is_dataclass(config) and not isinstance(config, type) else {}

    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "model_config": model_config,
        "state": asdict(state or TrainingState()),
        "meta": asdict(meta or CheckpointMeta()),
        "rng": rng_state(),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Scrittura atomica: prima il temporaneo, poi il rename.
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def load_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    scaler: Any = None,
    restore_rng: bool = True,
    map_location: str | torch.device | None = None,
) -> tuple[TrainingState, CheckpointMeta]:
    """Ricarica un checkpoint negli oggetti passati.

    Passare `model=None` permette di leggere solo i metadati, per ispezionare un file
    senza costruire la rete.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint assente: {path}")

    payload = torch.load(path, map_location=map_location or "cpu", weights_only=False)

    version = payload.get("meta", {}).get("schema_version", 0)
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"checkpoint di schema v{version}, atteso v{SCHEMA_VERSION}: "
            "il formato e cambiato, ricontrollare prima di usarlo"
        )

    if model is not None:
        model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and "scheduler" in payload:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and "scaler" in payload:
        scaler.load_state_dict(payload["scaler"])
    if restore_rng and "rng" in payload:
        restore_rng_state(payload["rng"])

    return TrainingState(**payload["state"]), CheckpointMeta(**payload["meta"])


def inspect_checkpoint(path: Path) -> str:
    """Riassunto leggibile di un checkpoint, senza caricare la rete."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    state = payload.get("state", {})
    meta = payload.get("meta", {})
    n_params = sum(v.numel() for v in payload["model"].values())
    size_mb = Path(path).stat().st_size / 1e6
    return (
        f"{Path(path).name}: {size_mb:.0f} MB, {n_params:,} parametri\n"
        f"  epoca {state.get('epoch')}, step {state.get('global_step'):,}, "
        f"{state.get('samples_seen', 0):,} campioni visti\n"
        f"  best val top-1: {100 * state.get('best_val_top1', 0):.2f}%\n"
        f"  commit {meta.get('commit')} — {meta.get('timestamp_utc')}\n"
        f"  tempo di training: {state.get('elapsed_seconds', 0) / 3600:.1f} h"
    )


class HistoryLog:
    """Registro append-only delle valutazioni, in JSONL.

    Separato dai checkpoint di proposito: i `.pt` sono grandi e si sovrascrivono, questo
    e piccolo e non si perde mai. Se un run va male, la storia di come ci e arrivato
    resta comunque leggibile — e si apre con qualunque cosa, anche senza PyTorch.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict[str, Any]) -> None:
        record = {"time": time.time(), **record}
        with open(self.path, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with open(self.path, encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]


__all__ = [
    "SCHEMA_VERSION",
    "CheckpointMeta",
    "HistoryLog",
    "TrainingState",
    "commit_hash",
    "inspect_checkpoint",
    "load_checkpoint",
    "restore_rng_state",
    "rng_state",
    "save_checkpoint",
]
