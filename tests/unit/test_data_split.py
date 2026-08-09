"""Gate 3 — split del dataset, dove vive la criticita #3.

Il test che conta e `test_zero_game_id_condivisi_fra_split`: se passa e uno split per
posizione viene introdotto per sbaglio, deve diventare rosso. E verificato esplicitamente
da `test_lo_split_per_posizione_verrebbe_intercettato`, che simula il bug e pretende che
il controllo lo veda.
"""

from __future__ import annotations

import chess
import pytest

from chessbot.data.pgn import WDL_DRAW, WDL_WIN, Sample
from chessbot.data.split import (
    SPLITS,
    TRAIN,
    SplitConfig,
    assign_split,
    position_key,
    split_with_stats,
    verify_no_leakage,
)

pytestmark = pytest.mark.fast


def make_samples(n_games: int, plies: int = 20, *, prefix: str = "g") -> list[Sample]:
    """Campioni sintetici: ogni partita ha le sue posizioni, tutte distinte.

    Le FEN sono finte ma univoche per (partita, ply): qui interessa la meccanica dello
    split, non la validita scacchistica. I test che hanno bisogno di posizioni vere
    usano `chess.Board`.
    """
    out: list[Sample] = []
    for g in range(n_games):
        for p in range(plies):
            out.append(
                Sample(
                    game_id=f"{prefix}{g:05d}",
                    fen=f"fen-{prefix}{g:05d}-{p:03d}",
                    move_uci="e2e4",
                    wdl=WDL_WIN if g % 2 else WDL_DRAW,
                    ply=p,
                )
            )
    return out


# --------------------------------------------------------------------------------------
# Criticita #3 — il criterio centrale del Gate 3
# --------------------------------------------------------------------------------------


def test_zero_game_id_condivisi_fra_split():
    """Criterio del Gate 3: nessuna partita puo comparire in due split."""
    assigned, _ = split_with_stats(make_samples(600))
    leaks = verify_no_leakage(assigned)
    assert leaks["game_id_condivisi"] == 0, f"leakage per partita: {leaks}"


def test_zero_fen_condivise_fra_split():
    """Criterio del Gate 3: nessuna posizione puo comparire in due split.

    Qui le partite condividono le prime posizioni, come accade davvero nelle aperture:
    e il caso che la sola deduplica per partita non risolverebbe.
    """
    samples: list[Sample] = []
    for g in range(400):
        for p in range(12):
            # I primi 4 ply sono in comune fra tutte le partite: teoria d'apertura.
            fen = f"apertura-{p:02d}" if p < 4 else f"fen-{g:05d}-{p:02d}"
            samples.append(
                Sample(game_id=f"g{g:05d}", fen=fen, move_uci="e2e4", wdl=WDL_DRAW, ply=p)
            )

    assigned, stats = split_with_stats(samples)
    leaks = verify_no_leakage(assigned)
    assert leaks["fen_condivise"] == 0, f"FEN condivise fra split: {leaks}"
    assert stats.dropped_dedup > 0, "la deduplica non ha scartato nulla, sospetto"


def test_dedup_ignora_i_contatori_della_fen():
    """Due FEN che differiscono solo per halfmove clock sono la STESSA posizione.

    Regressione di un leakage reale: sul dataset 2026-07 la deduplica confrontava le FEN
    complete, e 109 posizioni comparivano sia in train sia in val — stessi pezzi, stesso
    tratto, contatori diversi. Per la rete erano identiche.
    """
    base = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -"
    assert position_key(f"{base} 0 1") == position_key(f"{base} 12 34")

    # Due partite che finiscono in split diversi, sulla stessa posizione con clock
    # diverso: la seconda deve essere scartata dalla deduplica.
    samples = [
        Sample(game_id=f"g{i:05d}", fen=f"{base} {i} {i + 1}", move_uci="e2e4", wdl=WDL_DRAW, ply=0)
        for i in range(500)
    ]
    assigned, stats = split_with_stats(samples)
    leaks = verify_no_leakage(
        [
            (sp, Sample(s.game_id, position_key(s.fen), s.move_uci, s.wdl, s.ply))
            for sp, s in assigned
        ]
    )
    assert leaks["fen_condivise"] == 0
    assert stats.dropped_dedup > 0


def test_lo_split_per_posizione_verrebbe_intercettato():
    """Il controllo anti-leakage deve essere capace di FALLIRE.

    Un test che passa sempre non protegge nulla. Qui si simula esattamente il bug della
    criticita #3 — assegnare ogni POSIZIONE a caso invece di ogni partita — e si pretende
    che `verify_no_leakage` lo veda.
    """
    import random

    rng = random.Random(0)
    samples = make_samples(100)
    # Lo split sbagliato: per posizione, non per partita.
    assigned = [(rng.choice(SPLITS), s) for s in samples]

    leaks = verify_no_leakage(assigned)
    assert leaks["game_id_condivisi"] > 0, (
        "verify_no_leakage non ha visto uno split per posizione: il controllo e inutile"
    )


# --------------------------------------------------------------------------------------
# Determinismo e proporzioni
# --------------------------------------------------------------------------------------


def test_assegnazione_deterministica():
    """Stesso game_id e stesso seed danno sempre lo stesso split.

    Serve per poter ampliare il dataset senza rimescolare: una partita gia usata in
    training non deve poter finire in test dopo un secondo passaggio.
    """
    config = SplitConfig()
    for game_id in ("abc123", "zzz999", "g00042"):
        assert len({assign_split(game_id, config) for _ in range(50)}) == 1


def test_seed_diverso_cambia_l_assegnazione():
    a = SplitConfig(seed=1)
    b = SplitConfig(seed=2)
    ids = [f"g{i:05d}" for i in range(400)]
    diverse = sum(1 for gid in ids if assign_split(gid, a) != assign_split(gid, b))
    assert diverse > 0, "il seed non influenza lo split"


def test_proporzioni_rispettate():
    """95 / 2.5 / 2.5 con tolleranza: con 4000 partite la deviazione attesa e piccola."""
    config = SplitConfig()
    ids = [f"g{i:06d}" for i in range(4000)]
    counts = {s: 0 for s in SPLITS}
    for gid in ids:
        counts[assign_split(gid, config)] += 1

    assert abs(counts["train"] / 4000 - 0.95) < 0.02
    assert abs(counts["val"] / 4000 - 0.025) < 0.015
    assert abs(counts["test"] / 4000 - 0.025) < 0.015


def test_tutti_gli_split_sono_popolati():
    _, stats = split_with_stats(make_samples(800))
    for split in SPLITS:
        assert stats.per_split[split] > 0, f"split {split} vuoto"


def test_proporzioni_non_valide_rifiutate():
    with pytest.raises(ValueError):
        SplitConfig(train=0.9, val=0.2, test=0.2)
    with pytest.raises(ValueError):
        SplitConfig(train=1.0, val=0.0, test=0.0)


# --------------------------------------------------------------------------------------
# Capping delle aperture
# --------------------------------------------------------------------------------------


def test_capping_aperture():
    """Criterio del Gate 3: nessuna FEN oltre la soglia configurata."""
    cap = 10
    samples = [
        Sample(game_id=f"g{i:05d}", fen="posizione-comune", move_uci="e2e4", wdl=WDL_DRAW, ply=0)
        for i in range(200)
    ]
    # Tutte le partite nello stesso split, cosi la deduplica non interferisce.
    config = SplitConfig(max_occurrences_per_fen=cap, dedup_fen_across_splits=False)
    assigned, stats = split_with_stats(samples, config)

    tenute = sum(1 for _, s in assigned if s.fen == "posizione-comune")
    assert tenute == cap, f"capping non applicato: {tenute} occorrenze invece di {cap}"
    assert stats.dropped_capping == 200 - cap


def test_capping_disattivabile_alzando_la_soglia():
    samples = make_samples(20, plies=5)
    config = SplitConfig(max_occurrences_per_fen=10**9)
    assigned, stats = split_with_stats(samples, config)
    assert stats.dropped_capping == 0
    assert len(assigned) == 100


# --------------------------------------------------------------------------------------
# Statistiche
# --------------------------------------------------------------------------------------


def test_statistiche_coerenti():
    assigned, stats = split_with_stats(make_samples(300))
    assert stats.total == len(assigned)
    assert sum(stats.per_split.values()) == stats.total
    assert sum(stats.results.values()) == stats.total

    testo = stats.summary()
    for split in SPLITS:
        assert split in testo
    assert "esiti" in testo


def test_summary_non_esplode_su_dataset_vuoto():
    _, stats = split_with_stats([])
    assert "0" in stats.summary()


def test_le_partite_restano_intere():
    """Ogni partita deve finire tutta in uno split solo, mai spezzata."""
    assigned, _ = split_with_stats(make_samples(300))
    per_game: dict[str, set[str]] = {}
    for split, sample in assigned:
        per_game.setdefault(sample.game_id, set()).add(split)
    spezzate = {g: s for g, s in per_game.items() if len(s) > 1}
    assert not spezzate, f"partite spezzate fra split: {list(spezzate)[:3]}"


def test_lo_split_e_stabile_ampliando_il_dataset():
    """Aggiungere partite non deve spostare quelle gia assegnate."""
    config = SplitConfig()
    prima = {s.game_id: sp for sp, s in split_with_stats(make_samples(200), config)[0]}
    dopo = {s.game_id: sp for sp, s in split_with_stats(make_samples(400), config)[0]}
    for game_id, split in prima.items():
        assert dopo[game_id] == split, f"{game_id} ha cambiato split"


def test_una_partita_una_sola_board_reale():
    """Sanita: i campioni sintetici non nascondono un problema sulle FEN vere."""
    board = chess.Board()
    samples = []
    for i, move in enumerate(list(board.legal_moves)[:5]):
        samples.append(
            Sample(game_id="reale", fen=board.fen(), move_uci=move.uci(), wdl=WDL_WIN, ply=i)
        )
    assigned, _ = split_with_stats(samples)
    assert len({sp for sp, _ in assigned}) == 1
    assert assigned[0][0] in SPLITS
    assert TRAIN in SPLITS
