"""Split del dataset e deduplica (§2.1 del piano) — dove vive la criticita #3.

> ### Splittare per PARTITA, mai per posizione.
>
> Posizioni consecutive della stessa partita sono quasi identiche. Uno split casuale per
> posizione mette la stessa partita in train e validation: la validation loss risulta
> ottimisticamente falsa, credi di generalizzare e stai memorizzando.
>
> Il bug non crasha e non da alcun segnale. L'unico modo di escluderlo e un test che
> verifichi 0 game-id condivisi fra gli split — che e il primo criterio del Gate 3.

L'assegnazione allo split e **deterministica**, derivata da un hash del `game_id`, non
da un mescolamento casuale. Due conseguenze pratiche:

- rilanciare la conversione con lo stesso seed rimette ogni partita nello stesso split,
  anche se le partite arrivano in ordine diverso o il dataset viene ampliato
- non serve tenere in memoria la lista di tutti i game-id per fare lo split
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

from .pgn import Sample

TRAIN, VAL, TEST = "train", "val", "test"
SPLITS = (TRAIN, VAL, TEST)

# Precisione dell'assegnazione: l'hash viene ridotto a un intero in [0, RESOLUTION).
# 10^6 rende l'errore sulle proporzioni trascurabile rispetto al rumore statistico.
RESOLUTION = 1_000_000


@dataclass
class SplitConfig:
    """Proporzioni di §2.1. `by` esiste solo per essere documentato come non negoziabile."""

    train: float = 0.95
    val: float = 0.025
    test: float = 0.025
    seed: int = 1337
    dedup_fen_across_splits: bool = True
    max_occurrences_per_fen: int = 3000

    def __post_init__(self) -> None:
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"le proporzioni devono sommare a 1, non a {total}")
        if min(self.train, self.val, self.test) <= 0:
            raise ValueError("ogni split deve avere una proporzione positiva")


def position_key(fen: str) -> str:
    """Chiave di identita di una posizione: pezzi, tratto, arrocco, en passant.

    **Non** la FEN completa. Gli ultimi due campi sono l'halfmove clock e il numero di
    mossa: due posizioni identiche raggiunte per strade diverse hanno contatori diversi e
    FEN diverse, ma per la rete sono **la stessa posizione**.

    Deduplicare sulla FEN completa lascerebbe quindi passare del leakage: misurato sul
    dataset 2026-07, 109 posizioni comparivano sia in train sia in val proprio per questo
    — stessi pezzi, stesso tratto, halfmove clock diverso. Erano quasi tutte aperture,
    dove la stessa posizione si raggiunge con trasposizioni diverse.

    Il capping usa la stessa chiave, per lo stesso motivo: contare separatamente due
    varianti della stessa apertura vanificherebbe il limite.
    """
    return " ".join(fen.split()[:4])


def assign_split(game_id: str, config: SplitConfig) -> str:
    """Assegna una PARTITA a uno split, in modo deterministico e stabile.

    Si usa sha256 e non `hash()`: quest'ultimo e randomizzato per processo (PYTHONHASHSEED)
    e darebbe split diversi ad ogni esecuzione — un bug silenzioso che si manifesterebbe
    come leakage quando si aggiungono dati a un dataset gia usato per allenare.
    """
    digest = hashlib.sha256(f"{config.seed}:{game_id}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % RESOLUTION

    if bucket < config.train * RESOLUTION:
        return TRAIN
    if bucket < (config.train + config.val) * RESOLUTION:
        return VAL
    return TEST


@dataclass
class SplitStats:
    """Statistiche richieste dal Gate 3, da stampare e registrare."""

    per_split: Counter[str] = field(default_factory=Counter)
    games_per_split: dict[str, set[str]] = field(default_factory=lambda: {s: set() for s in SPLITS})
    results: Counter[int] = field(default_factory=Counter)
    with_eval: int = 0
    dropped_capping: int = 0
    dropped_dedup: int = 0
    fen_counter: Counter[str] = field(default_factory=Counter)

    @property
    def total(self) -> int:
        return sum(self.per_split.values())

    def summary(self) -> str:
        lines = [f"posizioni tenute: {self.total:,}"]
        for split in SPLITS:
            n = self.per_split[split]
            games = len(self.games_per_split[split])
            pct = 100 * n / self.total if self.total else 0
            lines.append(f"  {split:5s}: {n:>10,} posizioni ({pct:5.2f}%) da {games:,} partite")

        total_results = sum(self.results.values())
        if total_results:
            names = {0: "sconfitte bianco", 1: "patte", 2: "vittorie bianco"}
            dist = ", ".join(
                f"{names[k]} {100 * self.results[k] / total_results:.1f}%" for k in (2, 1, 0)
            )
            lines.append(f"  esiti: {dist}")

        if self.total:
            lines.append(f"  con [%eval]: {100 * self.with_eval / self.total:.1f}%")
        lines.append(f"  scartate per capping aperture: {self.dropped_capping:,}")
        lines.append(f"  scartate per deduplica fra split: {self.dropped_dedup:,}")

        top = self.fen_counter.most_common(5)
        if top:
            lines.append("  FEN piu frequenti:")
            for fen, count in top:
                lines.append(f"    {count:>7,}x  {fen[:60]}")
        return "\n".join(lines)


def split_samples(
    samples: Iterable[Sample],
    config: SplitConfig | None = None,
    *,
    stats: SplitStats | None = None,
) -> Iterator[tuple[str, Sample]]:
    """Assegna ogni campione a uno split, applicando capping e deduplica.

    Yield di `(split, sample)`, in streaming: sui dump veri il dataset non sta in
    memoria. L'ordine di ingresso e preservato — mescolare e compito del DataLoader.

    Due filtri, in quest'ordine:

    1. **capping delle aperture**: nessuna FEN oltre `max_occurrences_per_fen`. Senza,
       poche centinaia di posizioni iniziali si mangerebbero una fetta sproporzionata
       dei gradienti (§2.1).
    2. **deduplica fra split**: una FEN gia vista in uno split diverso viene scartata.
       Serve perche lo split e per partita, ma partite diverse condividono le stesse
       posizioni di apertura: senza questo passaggio le posizioni comuni comparirebbero
       sia in train sia in test.

    Args:
        stats: se fornito, viene popolato durante l'iterazione. E l'unico modo di
            ottenere le statistiche da un generatore: passare un oggetto e leggerlo
            dopo aver consumato l'iteratore.
    """
    config = config or SplitConfig()
    stats = stats if stats is not None else SplitStats()

    fen_counts: Counter[str] = Counter()
    fen_to_split: dict[str, str] = {}

    for sample in samples:
        split = assign_split(sample.game_id, config)
        key = position_key(sample.fen)

        if config.dedup_fen_across_splits:
            seen_in = fen_to_split.get(key)
            if seen_in is None:
                fen_to_split[key] = split
            elif seen_in != split:
                stats.dropped_dedup += 1
                continue

        fen_counts[key] += 1
        if fen_counts[key] > config.max_occurrences_per_fen:
            stats.dropped_capping += 1
            continue

        stats.per_split[split] += 1
        stats.games_per_split[split].add(sample.game_id)
        stats.results[sample.wdl] += 1
        stats.fen_counter[key] += 1
        if sample.eval_cp is not None:
            stats.with_eval += 1

        yield split, sample


def split_with_stats(
    samples: Iterable[Sample], config: SplitConfig | None = None
) -> tuple[list[tuple[str, Sample]], SplitStats]:
    """Come `split_samples`, ma materializza i risultati e restituisce le statistiche.

    Da usare quando il dataset sta in memoria (test, campioni piccoli). Sui dump veri si
    usa il generatore, passandogli uno `SplitStats` da leggere alla fine.
    """
    stats = SplitStats()
    out = list(split_samples(samples, config, stats=stats))
    return out, stats


def verify_no_leakage(assigned: Iterable[tuple[str, Sample]]) -> dict[str, int]:
    """Verifica i due criteri del Gate 3, in modo indipendente da come lo split e stato fatto.

    Non si fida della logica di assegnazione: ricostruisce gli insiemi dai dati prodotti
    e cerca le intersezioni. E il controllo che deve fallire se qualcuno un giorno
    "ottimizza" lo split rompendolo.

    Returns:
        Dizionario con `game_id_condivisi` e `fen_condivise`. Entrambi devono valere 0.
    """
    games: dict[str, set[str]] = {s: set() for s in SPLITS}
    fens: dict[str, set[str]] = {s: set() for s in SPLITS}

    for split, sample in assigned:
        games[split].add(sample.game_id)
        fens[split].add(sample.fen)

    shared_games = 0
    shared_fens = 0
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1 :]:
            shared_games += len(games[a] & games[b])
            shared_fens += len(fens[a] & fens[b])

    return {"game_id_condivisi": shared_games, "fen_condivise": shared_fens}
