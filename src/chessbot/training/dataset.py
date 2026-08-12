"""Dataset e DataLoader: dagli shard da 105 byte ai tensori 19x8x8 (Stadio 4).

E il pezzo che collega la pipeline dati alla rete. Su disco ci sono bitboard compressi;
la rete vuole tensori. La conversione avviene **al volo**, qui.

## Perche non pre-calcolare i tensori

    20.000.000 x 19 x 8 x 8 x 4 byte = 97 GB
    20.000.000 x 105 byte            =  2 GB

Quarantasei volte piu grande, e leggere 97 GB da disco ad ogni epoca sarebbe piu lento
che ricostruire i tensori in RAM. Il collo di bottiglia diventerebbe il disco invece
della GPU. Si paga invece un po' di CPU nei worker del DataLoader, che lavorano in
parallelo mentre la GPU calcola il batch precedente.

## Cosa produce

Ogni campione e un dizionario con quattro tensori:

| Chiave | Forma | Contenuto |
|---|---|---|
| `planes` | (19, 8, 8) float32 | la posizione, orientata dal lato che muove |
| `move_index` | scalare int64 | la mossa giocata, 0-4671 — etichetta della policy |
| `wdl` | scalare int64 | esito: 0 sconfitta, 1 patta, 2 vittoria (pov bianco) |
| `eval_cp` | scalare float32 | valutazione Stockfish, NaN se assente |
| `legal_mask` | (4672,) bool | quali indici sono mosse legali qui |

La maschera si calcola qui e non nel training loop perche richiede `python-chess`, che
gira su CPU: farlo nei worker lo mette in parallelo invece che in mezzo al passo di
ottimizzazione.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

from chessbot.data import SPLITS, read_shard, unpack_board
from chessbot.data.storage import NO_EVAL
from chessbot.encoding import BOARD_SIZE, N_MOVES, N_PLANES, encode_board, legal_move_mask

# Valutazione assente: NaN, cosi la loss puo escluderla con una maschera invece di
# trattare "nessun dato" come "valutazione zero", che sarebbe un'etichetta sbagliata.
EVAL_MISSING = float("nan")

# I centipedoni si dividono per questo prima di entrare nella loss: 600 cp e circa il
# punto oltre il quale una partita e decisa, e mappa il grosso dei valori in [-1, 1].
EVAL_SCALE = 600.0


def shard_paths(data_dir: Path, split: str) -> list[Path]:
    """Gli shard di uno split, in ordine deterministico."""
    if split not in SPLITS:
        raise ValueError(f"split sconosciuto: {split!r}, attesi {SPLITS}")
    return sorted((data_dir / split).glob("shard_*.npy"))


class ChessPositions(Dataset):
    """Accesso casuale a tutte le posizioni di uno split.

    Gli shard restano su disco e si aprono in memory-map: il sistema operativo pagina in
    RAM solo cio che serve davvero. Con 77 shard da 26 MB non si caricano 2 GB, si
    toccano le righe richieste.

    L'indice globale (0 .. N-1) si traduce in (shard, riga) con una ricerca binaria
    sugli offset cumulativi — nessuna struttura in memoria proporzionale al dataset.
    """

    def __init__(self, data_dir: Path, split: str, *, limit: int | None = None) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.paths = shard_paths(self.data_dir, split)
        if not self.paths:
            raise FileNotFoundError(f"nessuno shard in {self.data_dir / split}")

        # Quante righe ha ogni shard. Si legge l'header del .npy, non il contenuto.
        lengths = [len(np.load(p, mmap_mode="r")) for p in self.paths]
        self.offsets = np.cumsum([0] + lengths)
        self.total = int(self.offsets[-1])
        if limit is not None:
            self.total = min(self.total, limit)

        # Gli shard aperti vengono tenuti qui. Ogni worker ha il suo dizionario: un
        # memory-map condiviso fra processi darebbe letture corrotte.
        self._open: dict[int, np.ndarray] = {}
        self._worker_id: int | None = None

    def __len__(self) -> int:
        return self.total

    def _shard(self, index: int) -> np.ndarray:
        """Apre (o riusa) lo shard, ripartendo da zero se cambia il worker."""
        worker = get_worker_info()
        wid = worker.id if worker else None
        if wid != self._worker_id:
            self._open.clear()
            self._worker_id = wid
        if index not in self._open:
            self._open[index] = read_shard(self.paths[index])
        return self._open[index]

    def _locate(self, index: int) -> tuple[int, int]:
        """Indice globale -> (numero shard, riga dentro lo shard)."""
        if not 0 <= index < self.total:
            raise IndexError(f"indice {index} fuori da [0, {self.total})")
        shard = int(np.searchsorted(self.offsets, index, side="right") - 1)
        return shard, index - int(self.offsets[shard])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        shard_index, row_index = self._locate(index)
        row = self._shard(shard_index)[row_index]

        board = unpack_board(row["bitboards"], int(row["state"]), int(row["clock"]))

        # `repetitions=0`: lo storage non conserva il conteggio delle ripetizioni, e
        # ricostruirlo richiederebbe lo storico della partita, che non c'e. Il piano 18
        # resta quindi sempre a zero nel training supervisionato. E una perdita di
        # informazione accettata: le ripetizioni contano nell'MCTS (Stadio 5), dove lo
        # storico esiste, non nell'imitare una mossa da una posizione statica.
        planes = encode_board(board, repetitions=0)

        eval_raw = int(row["eval_cp"])
        eval_cp = EVAL_MISSING if eval_raw == NO_EVAL else eval_raw / EVAL_SCALE

        return {
            "planes": torch.from_numpy(planes),
            "move_index": torch.tensor(int(row["move_index"]), dtype=torch.long),
            "wdl": torch.tensor(int(row["wdl"]), dtype=torch.long),
            "eval_cp": torch.tensor(eval_cp, dtype=torch.float32),
            "legal_mask": torch.from_numpy(legal_move_mask(board)),
        }


class FixedPositions(Dataset):
    """Un numero fisso di posizioni tenute in RAM, per l'overfit test del Gate 4.

    Il test chiede ~100% di accuracy su 512 posizioni fisse. Devono essere le stesse ad
    ogni epoca e ad ogni riavvio, altrimenti non si sta misurando la memorizzazione: si
    ricalcolano una volta sola in `__init__` e restano li.
    """

    def __init__(self, source: ChessPositions, n: int, *, seed: int = 1337) -> None:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(source), size=min(n, len(source)), replace=False)
        self.samples = [source[int(i)] for i in sorted(indices)]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.samples[index]


def make_loader(
    dataset: Dataset,
    *,
    batch_size: int = 512,
    shuffle: bool = True,
    num_workers: int = 6,
    seed: int = 1337,
) -> torch.utils.data.DataLoader:
    """DataLoader con impostazioni riproducibili.

    > **Perche mescolare e obbligatorio.** Le posizioni sono salvate nell'ordine delle
    > partite: due righe consecutive sono la stessa posizione a un ply di distanza. Un
    > batch di righe consecutive avrebbe gradienti quasi identici, e il training sarebbe
    > instabile. In un batch da 512 servono 512 partite diverse (§2.2 del piano).
    """
    generator = torch.Generator()
    generator.manual_seed(seed)

    from chessbot.utils.seed import worker_init_fn

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,  # nel training scarta l'ultimo batch parziale
        generator=generator,
        worker_init_fn=worker_init_fn if num_workers else None,
        persistent_workers=num_workers > 0,
    )


def describe_batch(batch: dict[str, torch.Tensor]) -> str:
    """Riga di sanita su un batch, utile per capire cosa entra nella rete."""
    planes = batch["planes"]
    mask = batch["legal_mask"]
    with_eval = ~torch.isnan(batch["eval_cp"])
    return (
        f"batch {planes.shape[0]}: planes {tuple(planes.shape)} {planes.dtype}, "
        f"pezzi/posizione {planes[:, 0:12].sum(dim=(1, 2, 3)).mean():.1f}, "
        f"mosse legali/posizione {mask.sum(dim=1).float().mean():.1f}, "
        f"con eval {100 * with_eval.float().mean():.0f}%"
    )


def iter_batches(loader: torch.utils.data.DataLoader) -> Iterator[dict[str, torch.Tensor]]:
    """Alias esplicito: itera i batch. Esiste solo per rendere leggibile il training."""
    yield from loader


__all__ = [
    "BOARD_SIZE",
    "EVAL_SCALE",
    "N_MOVES",
    "N_PLANES",
    "ChessPositions",
    "FixedPositions",
    "describe_batch",
    "iter_batches",
    "make_loader",
    "shard_paths",
]
