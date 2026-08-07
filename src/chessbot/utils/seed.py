"""Seeding globale — base della riproducibilita (regola #2 di PIPELINE.md).

Un run che non sai riprodurre non e un risultato. Questo modulo mette sotto controllo
tutte le sorgenti di casualita usate dal progetto: `random`, numpy e torch (CPU e CUDA).
"""

from __future__ import annotations

import os
import random

DEFAULT_SEED = 1337


def set_seed(seed: int = DEFAULT_SEED, *, deterministic: bool = False) -> int:
    """Fissa il seed di tutte le sorgenti di casualita disponibili.

    Args:
        seed: valore da usare.
        deterministic: se True forza gli algoritmi deterministici di cuDNN. Costa
            velocita (le convoluzioni perdono l'autotuning), quindi va attivato per i
            test di riproducibilita, non per il training vero.

    Returns:
        Il seed applicato, comodo da scrivere nel checkpoint.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        # NPY002 raccomanda np.random.Generator, che e la scelta giusta per il codice
        # applicativo. Qui serve pero l'opposto: seedare lo stato GLOBALE legacy, che e
        # quello che usano i worker del DataLoader e le librerie di terze parti su cui non
        # abbiamo controllo. Un Generator locale non li raggiungerebbe.
        np.random.seed(seed)  # noqa: NPY002
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    return seed


def worker_init_fn(worker_id: int) -> None:
    """Da passare al DataLoader come `worker_init_fn`.

    Senza questo ogni worker eredita lo stesso stato RNG del processo padre e produce
    la stessa sequenza di numeri casuali: un bug silenzioso che riduce di fatto la
    diversita dei campioni senza dare alcun errore.
    """
    import torch

    base = torch.initial_seed() % 2**32
    set_seed(base + worker_id)
