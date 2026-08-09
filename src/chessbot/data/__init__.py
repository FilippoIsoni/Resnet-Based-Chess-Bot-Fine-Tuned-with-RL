"""Pipeline dati: da PGN a posizioni pronte per il training (Stadio 3, Fase 2).

> **La fase piu fragile del progetto.** Un bug qui non fa crashare niente: alleni su
> etichette sbagliate e te ne accorgi giorni dopo, guardando un'accuracy che non sale.

Il flusso, e dove sta il rischio in ognuno dei passi:

    PGN .zst  --filters--> partite valide  --pgn--> posizioni  --split--> train/val/test
                    |                          |                    |
              §2.1 Elo, TC              §2.2 rigioca         criticita #3
              esito, lunghezza          le mosse             split per PARTITA

    ... --storage--> shard .npy compatti (105 byte/posizione, non 4.8 KB)

Uso tipico:

    from chessbot.data import FilterConfig, SplitConfig, iter_samples, split_samples

    samples = iter_samples(Path("data/raw/dump.pgn.zst"), FilterConfig())
    for split, sample in split_samples(samples, SplitConfig()):
        ...
"""

from .filters import (
    FilterConfig,
    FilterStats,
    accept_headers,
    average_elo,
    category,
    estimated_duration,
)
from .pgn import (
    WDL_DRAW,
    WDL_LOSS,
    WDL_WIN,
    Sample,
    iter_games,
    iter_samples,
    samples_from_game,
)
from .split import (
    SPLITS,
    TEST,
    TRAIN,
    VAL,
    SplitConfig,
    SplitStats,
    assign_split,
    position_key,
    split_samples,
    split_with_stats,
    verify_no_leakage,
)
from .storage import (
    SAMPLE_DTYPE,
    bytes_per_sample,
    encode_samples,
    iter_shard_boards,
    pack_board,
    read_shard,
    unpack_board,
    write_shard,
)

__all__ = [
    "SAMPLE_DTYPE",
    "SPLITS",
    "TEST",
    "TRAIN",
    "VAL",
    "WDL_DRAW",
    "WDL_LOSS",
    "WDL_WIN",
    "FilterConfig",
    "FilterStats",
    "Sample",
    "SplitConfig",
    "SplitStats",
    "accept_headers",
    "assign_split",
    "average_elo",
    "bytes_per_sample",
    "category",
    "encode_samples",
    "estimated_duration",
    "iter_games",
    "iter_samples",
    "iter_shard_boards",
    "pack_board",
    "position_key",
    "read_shard",
    "samples_from_game",
    "split_samples",
    "split_with_stats",
    "unpack_board",
    "verify_no_leakage",
    "write_shard",
]
