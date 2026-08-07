"""Test del seeding globale.

Verifica la regola #2 di PIPELINE.md: un run che non sai riprodurre non e un risultato.
Questi test sono anche la prova che l'infrastruttura di check funziona end-to-end prima
che esista codice di dominio.
"""

from __future__ import annotations

import random

import pytest

from chessbot.utils.seed import DEFAULT_SEED, set_seed


@pytest.mark.fast
def test_set_seed_restituisce_il_seed() -> None:
    assert set_seed(42) == 42
    assert set_seed() == DEFAULT_SEED


@pytest.mark.fast
def test_random_riproducibile() -> None:
    set_seed(1234)
    prima = [random.random() for _ in range(10)]
    set_seed(1234)
    dopo = [random.random() for _ in range(10)]
    assert prima == dopo


@pytest.mark.fast
def test_seed_diversi_danno_sequenze_diverse() -> None:
    """Controprova: senza questo, un set_seed rotto che azzera tutto passerebbe comunque."""
    set_seed(1)
    a = [random.random() for _ in range(10)]
    set_seed(2)
    b = [random.random() for _ in range(10)]
    assert a != b


@pytest.mark.fast
def test_numpy_riproducibile() -> None:
    np = pytest.importorskip("numpy")

    set_seed(7)
    a = np.random.rand(10)
    set_seed(7)
    b = np.random.rand(10)
    assert np.array_equal(a, b)


@pytest.mark.fast
def test_torch_riproducibile() -> None:
    torch = pytest.importorskip("torch")

    set_seed(99)
    a = torch.rand(10)
    set_seed(99)
    b = torch.rand(10)
    assert torch.equal(a, b)
