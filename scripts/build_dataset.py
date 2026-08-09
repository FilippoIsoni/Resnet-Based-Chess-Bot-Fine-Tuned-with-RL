"""Converte un dump PGN in shard pronti per il training (Stadio 3).

    python scripts/build_dataset.py --input data/raw/lichess_db_standard_rated_2026-07.pgn.zst
    python scripts/build_dataset.py --input ... --target-positions 5000000
    python scripts/build_dataset.py --input ... --max-games 2000 --out data/processed/prova

Produce in `data/processed/`:

    train/shard_00000.npy, train/shard_00001.npy, ...
    val/shard_00000.npy
    test/shard_00000.npy
    manifest.json          <- statistiche, config e commit hash

Il manifest e la parte che il Gate 3 legge: contiene i conteggi per split, la
distribuzione degli esiti, le FEN piu frequenti e il numero di game-id e FEN condivise
fra split, che devono essere entrambi 0 (criticita #3).

Single-process di proposito: il collo di bottiglia e il parsing di python-chess, e su
questo dump la conversione resta nell'ordine dell'ora. La parallelizzazione con
`multiprocessing.Pool` e prevista da §2.6 ma e ottimizzazione — regola #4, prima si
misura.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from chessbot.data import (  # noqa: E402
    SPLITS,
    FilterConfig,
    FilterStats,
    Sample,
    SplitConfig,
    SplitStats,
    iter_samples,
    split_samples,
    write_shard,
)
from chessbot.utils.seed import set_seed  # noqa: E402

# Posizioni per shard. 250k x 105 byte = ~26 MB: abbastanza grande da non moltiplicare i
# file, abbastanza piccolo da rileggerne uno senza pesare sulla RAM.
SHARD_SIZE = 250_000


def commit_hash() -> str:
    p = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
    )
    return p.stdout.strip() or "sconosciuto"


class ShardWriter:
    """Accumula campioni per split e scrive uno shard ogni SHARD_SIZE."""

    def __init__(self, out_dir: Path, shard_size: int = SHARD_SIZE) -> None:
        self.out_dir = out_dir
        self.shard_size = shard_size
        self.buffers: dict[str, list[Sample]] = {s: [] for s in SPLITS}
        self.counts: Counter[str] = Counter()
        self.shards: Counter[str] = Counter()

    def add(self, split: str, sample: Sample) -> None:
        buf = self.buffers[split]
        buf.append(sample)
        if len(buf) >= self.shard_size:
            self._flush(split)

    def _flush(self, split: str) -> None:
        buf = self.buffers[split]
        if not buf:
            return
        index = self.shards[split]
        path = self.out_dir / split / f"shard_{index:05d}.npy"
        written = write_shard(path, buf)
        self.counts[split] += written
        self.shards[split] += 1
        buf.clear()

    def close(self) -> None:
        for split in SPLITS:
            self._flush(split)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True, help="dump .pgn o .pgn.zst")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "processed")
    ap.add_argument(
        "--target-positions",
        type=int,
        default=None,
        help="ferma la conversione raggiunte N posizioni (default: tutto il dump)",
    )
    ap.add_argument("--max-games", type=int, default=None, help="limite di partite accettate")
    ap.add_argument("--min-elo", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    args = ap.parse_args()

    if not args.input.exists():
        print(f"file non trovato: {args.input}")
        return 1

    set_seed(args.seed)
    filter_config = FilterConfig(min_elo=args.min_elo)
    split_config = SplitConfig(seed=args.seed)
    filter_stats = FilterStats()
    split_stats = SplitStats()

    print(f"input : {args.input} ({args.input.stat().st_size / 1e9:.1f} GB)")
    print(f"output: {args.out}")
    print(f"filtri: Elo >= {args.min_elo}, {', '.join(filter_config.time_controls)}")
    if args.target_positions:
        print(f"target: {args.target_positions:,} posizioni")
    print()

    writer = ShardWriter(args.out, args.shard_size)
    samples = iter_samples(args.input, filter_config, stats=filter_stats, max_games=args.max_games)

    started = time.perf_counter()
    last_report = 0.0
    kept = 0

    try:
        for split, sample in split_samples(samples, split_config, stats=split_stats):
            writer.add(split, sample)
            kept += 1

            now = time.perf_counter()
            if now - last_report >= 30:
                rate = kept / (now - started)
                pct = (
                    f" ({100 * kept / args.target_positions:.1f}%)" if args.target_positions else ""
                )
                print(
                    f"  {kept:,} posizioni{pct} da {filter_stats.kept:,} partite "
                    f"({filter_stats.total:,} lette) — {rate:.0f} pos/s",
                    flush=True,
                )
                last_report = now

            if args.target_positions and kept >= args.target_positions:
                print(f"  raggiunto il target di {args.target_positions:,} posizioni")
                break
    except KeyboardInterrupt:
        print("\ninterrotto — scrivo gli shard parziali")
    finally:
        writer.close()

    elapsed = time.perf_counter() - started
    print()
    print(filter_stats.summary())
    print()
    print(split_stats.summary())
    print()
    print(f"tempo: {elapsed / 60:.1f} min ({kept / max(elapsed, 1e-9):.0f} posizioni/s)")

    manifest = {
        "timestamp_utc": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
        "commit": commit_hash(),
        "input": str(args.input.name),
        "seed": args.seed,
        "filters": {
            "min_elo": args.min_elo,
            "time_controls": list(filter_config.time_controls),
            "min_moves": filter_config.min_moves,
            "exclude_timeout_in_non_lost": filter_config.exclude_timeout_in_non_lost,
        },
        "split": {
            "train": split_config.train,
            "val": split_config.val,
            "test": split_config.test,
            "by": "game",
            "dedup_fen_across_splits": split_config.dedup_fen_across_splits,
            "max_occurrences_per_fen": split_config.max_occurrences_per_fen,
        },
        "games_read": filter_stats.total,
        "games_kept": filter_stats.kept,
        "rejected": dict(filter_stats.rejected),
        "positions": dict(writer.counts),
        "shards": dict(writer.shards),
        "games_per_split": {s: len(split_stats.games_per_split[s]) for s in SPLITS},
        "results": {str(k): v for k, v in split_stats.results.items()},
        "with_eval": split_stats.with_eval,
        "dropped_capping": split_stats.dropped_capping,
        "dropped_dedup": split_stats.dropped_dedup,
        "top_fens": [
            {"fen": fen, "count": count} for fen, count in split_stats.fen_counter.most_common(20)
        ],
        "elapsed_seconds": round(elapsed, 1),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"manifest: {manifest_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
