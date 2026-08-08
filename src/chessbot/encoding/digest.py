"""Hash stabile fra piattaforme di un tensore di encoding.

Vive nel package e non negli script perche lo usano entrambi: `scripts/gen_golden.py`
per scrivere il riferimento e i test per verificarlo. Due implementazioni separate della
stessa funzione finirebbero prima o poi per divergere, ed e esattamente il tipo di
differenza silenziosa contro cui e costruito il Gate 1.

> Stadio 0-bis — Il digest deve coincidere su Windows e macOS. Per questo i valori
> vengono quantizzati a interi (via floating point le due macchine possono differire
> nell'ultimo bit) e serializzati con endianness esplicita, che altrimenti dipenderebbe
> dalla CPU.
"""

from __future__ import annotations

import hashlib

import numpy as np

# I valori dell'encoder sono 0, 1 o frazioni con pochi decimali significativi: 10^4 e
# abbondante per distinguerli tutti e sufficiente a cancellare il rumore floating point.
QUANTIZATION = 10_000


def tensor_digest(planes: np.ndarray) -> str:
    """SHA-256 del tensore, identico su ogni piattaforma."""
    quantized = np.rint(planes.astype(np.float64) * QUANTIZATION).astype(">i4")
    return hashlib.sha256(quantized.tobytes()).hexdigest()
