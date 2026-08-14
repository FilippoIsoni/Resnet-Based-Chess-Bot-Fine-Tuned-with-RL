"""Stadio 6 — Expert Iteration: self-play parallelo, gating, loss RL.

Test veloci su reti finte. Il ciclo vero costa ore e si lancia con `scripts/run_rl.py`.
"""

from __future__ import annotations

import itertools
import random
from pathlib import Path

import chess
import numpy as np
import pytest
import torch

from chessbot.encoding import N_MOVES
from chessbot.model.network import ChessNet, NetworkConfig
from chessbot.search import SearchConfig
from chessbot.search.openings import book_diversity, sample_openings
from chessbot.search.parallel import GameRecord, self_play_batch, value_target
from chessbot.training.expert_iteration import (
    IterationResult,
    ReplayBuffer,
    RLConfig,
    build_rl_optimizer,
    load_rl_config,
    records_to_tensors,
    rl_loss,
    should_stop,
)
from chessbot.training.gating import SprtTest, SprtVerdict, elo_to_score

pytestmark = pytest.mark.fast

ROOT = Path(__file__).resolve().parents[2]


class UniformEvaluator:
    def evaluate(self, boards: list[chess.Board]) -> list[tuple[dict, float]]:
        out = []
        for board in boards:
            moves = list(board.legal_moves)
            uniform = 1.0 / max(len(moves), 1)
            out.append(({m: uniform for m in moves}, 0.0))
        return out


# --------------------------------------------------------------------------------------
# Self-play parallelo — criticita #8
# --------------------------------------------------------------------------------------


def test_self_play_produce_record_completi():
    records, stats = self_play_batch(
        UniformEvaluator(), n_games=4, config=SearchConfig(simulations=8), max_plies=12
    )
    assert stats.games == 4
    assert records

    for r in records:
        chess.Board(r.fen)  # deve essere una FEN valida
        assert r.visit_counts, "nessuna visita registrata"
        assert r.result is not None, "il risultato deve essere riempito a fine partita"
        assert -1.0 <= r.result <= 1.0
        assert -1.0 <= r.q_root <= 1.0


def test_il_batching_fra_partite_raggruppa():
    """Il criterio del Gate 6: piu partite in parallelo -> batch piu grandi."""
    _, una = self_play_batch(
        UniformEvaluator(), n_games=1, config=SearchConfig(simulations=8), max_plies=8
    )
    _, otto = self_play_batch(
        UniformEvaluator(), n_games=8, config=SearchConfig(simulations=8), max_plies=8
    )
    assert otto.avg_batch > una.avg_batch * 3, (
        f"batch medio {otto.avg_batch:.1f} con 8 partite contro {una.avg_batch:.1f} con 1"
    )


def test_le_mosse_del_self_play_sono_legali():
    records, _ = self_play_batch(
        UniformEvaluator(), n_games=3, config=SearchConfig(simulations=8), max_plies=10
    )
    for r in records:
        board = chess.Board(r.fen)
        for move in r.visit_counts:
            assert move in board.legal_moves, f"{move} illegale in {r.fen}"


def test_il_risultato_alterna_col_tratto():
    """z e dal punto di vista di chi muoveva: due posizioni consecutive della stessa
    partita devono avere segno opposto (salvo le patte)."""
    records, _ = self_play_batch(
        UniformEvaluator(), n_games=2, config=SearchConfig(simulations=8), max_plies=20
    )
    # Raggruppa per partita usando il colore alternato: posizioni consecutive con
    # tratto diverso devono avere risultato opposto.
    for a, b in itertools.pairwise(records):
        if a.to_play != b.to_play and a.result is not None and b.result is not None:
            if a.result != 0.0:
                assert a.result == -b.result, "z non alterna col tratto"
            break


def test_le_aperture_diversificano_le_partite():
    """Criticita #7: partendo da posizioni diverse, le partite non si somigliano."""
    aperture = [
        chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"),
        chess.Board("rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"),
    ]
    records, _ = self_play_batch(
        UniformEvaluator(),
        n_games=4,
        config=SearchConfig(simulations=8),
        openings=aperture,
        max_plies=8,
    )
    prime = {r.fen for r in records if chess.Board(r.fen).fullmove_number <= 2}
    assert len(prime) >= 2, "tutte le partite partono dalla stessa posizione"


# --------------------------------------------------------------------------------------
# Target del valore — §5.5
# --------------------------------------------------------------------------------------


def test_target_misto_fra_q_e_risultato():
    r = GameRecord(fen=chess.STARTING_FEN, visit_counts={}, q_root=0.5, to_play=chess.WHITE)
    r.result = -1.0

    assert value_target(r, 0.0) == pytest.approx(-1.0), "alpha=0 -> solo il risultato"
    assert value_target(r, 1.0) == pytest.approx(0.5), "alpha=1 -> solo Q"
    assert value_target(r, 0.4) == pytest.approx(0.4 * 0.5 + 0.6 * -1.0)


# --------------------------------------------------------------------------------------
# Gating con SPRT — §5.2
# --------------------------------------------------------------------------------------


def test_sprt_non_decide_dopo_una_partita():
    """Regressione di un bug grave.

    > La prima formula della LLR usava la varianza CAMPIONARIA del punteggio. Con una
    > sola partita la varianza e zero, il clamp a 1e-6 faceva esplodere la LLR a
    > **17.313** contro una soglia di 2,94, e il test decideva dopo il primo incontro.
    > Misurato su quattro scenari a forza nota: tutti decisi subito, tre su quattro
    > sbagliati. Una rete identica veniva promossa, una migliore rifiutata.
    """
    test = SprtTest()
    verdict = test.update(1.0)
    assert verdict == SprtVerdict.CONTINUE, f"deciso dopo 1 partita con LLR {test.llr}"
    assert abs(test.llr) < 5.0, f"LLR {test.llr} fuori scala dopo una partita"


def test_sprt_promuove_una_rete_nettamente_migliore():
    rng = random.Random(7)
    p = elo_to_score(80)
    promosse = 0
    for seed in range(20):
        r = random.Random(seed * 31 + 3)
        test = SprtTest()
        for _ in range(200):
            outcome = 0.5 if r.random() < 0.3 else (1.0 if r.random() < p else 0.0)
            if test.update(outcome) != SprtVerdict.CONTINUE:
                break
        if test.verdict() == SprtVerdict.ACCEPT:
            promosse += 1
    assert promosse >= 14, f"solo {promosse}/20 promozioni per una rete a +80 Elo"
    assert rng  # silenzia il linter sul generatore non usato


def test_sprt_rifiuta_una_rete_peggiore():
    p = elo_to_score(-80)
    promosse = 0
    for seed in range(20):
        r = random.Random(seed * 17 + 5)
        test = SprtTest()
        for _ in range(200):
            outcome = 0.5 if r.random() < 0.3 else (1.0 if r.random() < p else 0.0)
            if test.update(outcome) != SprtVerdict.CONTINUE:
                break
        if test.verdict() == SprtVerdict.ACCEPT:
            promosse += 1
    assert promosse == 0, f"{promosse}/20 promozioni per una rete a -80 Elo"


def test_sprt_raramente_promuove_una_rete_identica():
    """Il tasso di falsi positivi deve stare vicino ad alpha (5%)."""
    promosse = 0
    for seed in range(40):
        r = random.Random(seed * 13 + 1)
        test = SprtTest()
        for _ in range(200):
            outcome = 0.5 if r.random() < 0.3 else (1.0 if r.random() < 0.5 else 0.0)
            if test.update(outcome) != SprtVerdict.CONTINUE:
                break
        if test.verdict() == SprtVerdict.ACCEPT:
            promosse += 1
    assert promosse <= 8, f"{promosse}/40 falsi positivi, troppi per alpha=0.05"


def test_sprt_solo_patte_non_decide():
    test = SprtTest()
    for _ in range(50):
        test.update(0.5)
    assert test.llr == 0.0, "le patte non portano informazione sulla differenza di forza"


def test_le_soglie_sprt_derivano_da_alpha_e_beta():
    test = SprtTest(alpha=0.05, beta=0.05)
    assert test.upper_bound == pytest.approx(np.log(0.95 / 0.05), abs=1e-6)
    assert test.lower_bound == pytest.approx(np.log(0.05 / 0.95), abs=1e-6)


# --------------------------------------------------------------------------------------
# Loss RL
# --------------------------------------------------------------------------------------


def tiny_net() -> ChessNet:
    return ChessNet(NetworkConfig(channels=8, blocks=1, value_hidden=8))


def fake_records(n: int = 6) -> list[GameRecord]:
    board = chess.Board()
    moves = list(board.legal_moves)[:5]
    out = []
    for i in range(n):
        r = GameRecord(
            fen=chess.STARTING_FEN,
            visit_counts={m: 10 + j for j, m in enumerate(moves)},
            q_root=0.1 * i,
            to_play=chess.WHITE,
        )
        r.result = 1.0 if i % 2 else -1.0
        out.append(r)
    return out


def test_records_to_tensors():
    config = RLConfig()
    planes, policy, values, masks = records_to_tensors(fake_records(4), config)

    assert planes.shape == (4, 19, 8, 8)
    assert policy.shape == (4, N_MOVES)
    assert values.shape == (4,)
    assert masks.shape == (4, N_MOVES)
    # Il target della policy e una distribuzione.
    assert policy.sum(dim=1).allclose(torch.ones(4), atol=1e-5)
    # Solo le mosse presenti nei visit_counts sono marcate legali.
    assert masks.sum(dim=1).tolist() == [5, 5, 5, 5]


def test_la_loss_rl_e_finita_e_derivabile():
    config = RLConfig()
    net = tiny_net()
    planes, policy, values, masks = records_to_tensors(fake_records(4), config)

    policy_logits, wdl_logits = net(planes)
    loss, _parts = rl_loss(policy_logits, wdl_logits, policy, values, masks, None, config)

    assert torch.isfinite(loss)
    assert loss.requires_grad
    loss.backward()
    senza = [n for n, p in net.named_parameters() if p.requires_grad and p.grad is None]
    assert not senza, f"parametri senza gradiente: {senza}"


def test_la_penalita_kl_aumenta_la_loss():
    """Con una policy di riferimento diversa, la KL deve pesare."""
    config = RLConfig(kl_penalty_weight=1.0)
    net = tiny_net()
    planes, policy, values, masks = records_to_tensors(fake_records(4), config)

    policy_logits, wdl_logits = net(planes)
    reference = torch.randn_like(policy_logits) * 3

    _, senza_kl = rl_loss(
        policy_logits, wdl_logits, policy, values, masks, None, RLConfig(kl_penalty_weight=0.0)
    )
    _, con_kl = rl_loss(policy_logits, wdl_logits, policy, values, masks, reference, config)

    assert con_kl["kl"] > 0
    assert con_kl["total"] > senza_kl["total"]


def test_optimizer_e_sgd_non_adam():
    """§5.2: SGD e piu stabile di Adam sotto distribution shift."""
    optimizer = build_rl_optimizer(tiny_net(), RLConfig())
    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.param_groups[0]["momentum"] == pytest.approx(0.9)


# --------------------------------------------------------------------------------------
# Replay buffer
# --------------------------------------------------------------------------------------


def test_il_buffer_tiene_una_finestra():
    buffer = ReplayBuffer(max_iterations=3)
    for _ in range(5):
        buffer.add(fake_records(2))
    assert len(buffer.chunks) == 3, "la finestra non e rispettata"
    assert len(buffer) == 6


# --------------------------------------------------------------------------------------
# Criteri di stop — criticita #9
# --------------------------------------------------------------------------------------


def make_result(i: int, *, accepted: bool = False, elo: float = 0.0) -> IterationResult:
    return IterationResult(
        iteration=i,
        games=10,
        positions=100,
        draw_rate=0.3,
        gating_accepted=accepted,
        gating_summary="",
        elo_estimate=elo,
        losses={},
        elapsed_seconds=1.0,
    )


def test_stop_al_tetto_di_iterazioni():
    config = RLConfig(max_iterations=5)
    stop, reason = should_stop([make_result(i) for i in range(5)], config)
    assert stop and "tetto duro" in reason


def test_stop_se_il_gating_non_passa_mai():
    config = RLConfig(abort_if_no_gating_pass_within=4, max_iterations=40)
    stop, reason = should_stop([make_result(i) for i in range(4)], config)
    assert stop and "gating mai passato" in reason


def test_non_ferma_se_il_gating_passa():
    config = RLConfig(abort_if_no_gating_pass_within=4, max_iterations=40)
    results = [make_result(i, accepted=(i == 1), elo=20) for i in range(4)]
    stop, _ = should_stop(results, config)
    assert not stop


def test_stop_se_il_guadagno_cumulativo_e_basso():
    config = RLConfig(abort_elo_check_at=5, abort_if_cumulative_elo_below=50, max_iterations=40)
    results = [make_result(i, accepted=True, elo=5) for i in range(5)]
    stop, reason = should_stop(results, config)
    assert stop and "cumulativo" in reason


def test_non_ferma_se_il_guadagno_e_buono():
    config = RLConfig(abort_elo_check_at=5, abort_if_cumulative_elo_below=50, max_iterations=40)
    results = [make_result(i, accepted=True, elo=30) for i in range(5)]
    stop, _ = should_stop(results, config)
    assert not stop


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------


def test_carica_i_criteri_di_stop_dal_file():
    """Criticita #9: i criteri stanno in configs/rl.yaml, non nel codice."""
    config = load_rl_config(ROOT / "configs" / "rl.yaml")
    assert config.max_iterations == 40
    assert config.abort_if_no_gating_pass_within == 10
    assert config.abort_if_cumulative_elo_below == 50
    assert config.kl_penalty_weight == pytest.approx(0.075)
    assert config.alpha_q_root == pytest.approx(0.4)


# --------------------------------------------------------------------------------------
# Libro d'aperture — criticita #7
# --------------------------------------------------------------------------------------


def test_sample_openings_pesca_dal_libro():
    book = [chess.Board(), chess.Board("8/8/8/4k3/8/8/8/4K3 w - - 0 1")]
    rng = np.random.default_rng(0)
    sampled = sample_openings(book, 10, rng)
    assert len(sampled) == 10
    assert all(isinstance(b, chess.Board) for b in sampled)


def test_sample_openings_senza_libro_usa_la_posizione_iniziale():
    sampled = sample_openings([], 3, np.random.default_rng(0))
    assert all(b.fen() == chess.STARTING_FEN for b in sampled)


def test_book_diversity_classifica_le_aperture():
    book = [
        chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"),
        chess.Board("rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2"),
    ]
    families = book_diversity(book)
    assert "1.e4" in families
    assert "1.d4" in families
