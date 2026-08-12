"""Stadio 4 — rete, loss, checkpoint.

I test qui girano su tensori sintetici e reti piccole: verificano la meccanica, non la
qualita dell'apprendimento. Quella la misura l'overfit test su dati veri.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from chessbot.encoding import BOARD_SIZE, N_MOVES, N_PLANES
from chessbot.model.network import ChessNet, NetworkConfig, masked_policy_logits
from chessbot.training.checkpoint import (
    CheckpointMeta,
    HistoryLog,
    TrainingState,
    load_checkpoint,
    save_checkpoint,
)
from chessbot.training.loop import TrainConfig, build_optimizer, build_scheduler
from chessbot.training.loss import (
    accuracy_metrics,
    compute_loss,
    masked_cross_entropy,
)

pytestmark = pytest.mark.fast


def tiny_net() -> ChessNet:
    """Rete minuscola: i test verificano la meccanica, non servono 12M parametri."""
    return ChessNet(NetworkConfig(channels=8, blocks=1, value_hidden=8))


def fake_batch(size: int = 4, n_legal: int = 20, *, with_eval: bool = True) -> dict:
    mask = torch.zeros(size, N_MOVES, dtype=torch.bool)
    mask[:, :n_legal] = True
    return {
        "planes": torch.randn(size, N_PLANES, BOARD_SIZE, BOARD_SIZE),
        "move_index": torch.randint(0, n_legal, (size,)),
        "wdl": torch.randint(0, 3, (size,)),
        "eval_cp": torch.randn(size) if with_eval else torch.full((size,), float("nan")),
        "legal_mask": mask,
    }


# --------------------------------------------------------------------------------------
# Rete
# --------------------------------------------------------------------------------------


def test_forma_delle_uscite():
    net = tiny_net()
    policy, wdl = net(torch.randn(3, N_PLANES, BOARD_SIZE, BOARD_SIZE))
    assert policy.shape == (3, N_MOVES)
    assert wdl.shape == (3, 3)


def test_forma_di_ingresso_sbagliata_rifiutata():
    net = tiny_net()
    with pytest.raises(ValueError):
        net(torch.randn(3, 12, 8, 8))
    with pytest.raises(ValueError):
        net(torch.randn(N_PLANES, BOARD_SIZE, BOARD_SIZE))


def test_il_blocco_residuo_preserva_la_forma():
    from chessbot.model.network import ResidualBlock

    block = ResidualBlock(16)
    x = torch.randn(2, 16, 8, 8)
    assert block(x).shape == x.shape


def test_configurazione_non_valida_rifiutata():
    with pytest.raises(ValueError):
        NetworkConfig(blocks=0)
    with pytest.raises(ValueError):
        NetworkConfig(channels=0)


def test_i_gradienti_arrivano_a_tutti_i_parametri():
    """Se un parametro non riceve gradiente, e scollegato dalla loss e non impara mai."""
    net = tiny_net()
    batch = fake_batch()
    policy, wdl = net(batch["planes"])
    compute_loss(policy, wdl, batch).total.backward()

    senza = [n for n, p in net.named_parameters() if p.requires_grad and p.grad is None]
    assert not senza, f"parametri senza gradiente: {senza}"


# --------------------------------------------------------------------------------------
# Mascheratura — criticita #12
# --------------------------------------------------------------------------------------


def test_la_maschera_azzera_le_mosse_illegali():
    logits = torch.randn(2, N_MOVES)
    mask = torch.zeros(2, N_MOVES, dtype=torch.bool)
    mask[:, :10] = True

    probs = torch.softmax(masked_policy_logits(logits, mask), dim=1)
    assert probs[:, 10:].sum().item() == pytest.approx(0.0, abs=1e-6)
    assert probs[:, :10].sum(dim=1).tolist() == pytest.approx([1.0, 1.0], abs=1e-5)


def test_maschera_di_forma_sbagliata_rifiutata():
    with pytest.raises(ValueError):
        masked_policy_logits(torch.randn(2, N_MOVES), torch.ones(3, N_MOVES, dtype=torch.bool))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_la_maschera_funziona_in_precisione_mista(dtype):
    """Regressione: con `-1e9` fisso, in fp16 si otteneva
    `RuntimeError: value cannot be converted to type c10::Half without overflow`.

    Il massimo rappresentabile in fp16 e 65504, quindi -1e9 va in overflow. Con la
    precisione mista attiva — il default sulla 3050 — i logit arrivano alla mascheratura
    in fp16, e il training crashava al primo batch. I test unitari non lo vedevano
    perche girano tutti in fp32.
    """
    logits = torch.randn(2, N_MOVES, dtype=dtype)
    mask = torch.zeros(2, N_MOVES, dtype=torch.bool)
    mask[:, :10] = True

    masked = masked_policy_logits(logits, mask)
    assert masked.dtype == dtype
    assert torch.isfinite(masked).all(), "la sentinella ha prodotto inf o NaN"

    probs = torch.softmax(masked.float(), dim=1)
    assert probs[:, 10:].sum().item() == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_la_loss_regge_la_precisione_mista(dtype):
    """La catena completa mascheratura -> log_softmax -> loss, nei due dtype."""
    batch = fake_batch()
    policy = torch.randn(4, N_MOVES, dtype=dtype)
    wdl = torch.randn(4, 3, dtype=dtype)

    masked = masked_policy_logits(policy, batch["legal_mask"])
    loss = masked_cross_entropy(
        masked, batch["move_index"], batch["legal_mask"], label_smoothing=0.05
    )
    assert torch.isfinite(loss), f"loss non finita in {dtype}"
    assert loss.item() < 100, f"loss {loss.item()} in {dtype}"
    assert wdl.shape == (4, 3)


# --------------------------------------------------------------------------------------
# Loss — regressione del bug del label smoothing
# --------------------------------------------------------------------------------------


def test_lo_smoothing_non_esplode_sulle_mosse_illegali():
    """Regressione: `F.cross_entropy(label_smoothing=...)` dava 49.683.864 invece di 3,4.

    Il label smoothing di PyTorch distribuisce massa su TUTTE le classi, comprese le
    ~4642 illegali portate a -1e9 dalla mascheratura. Il gradiente sui logit legali
    restava corretto — l'overfit test passava lo stesso — ma la loss era illeggibile e
    schiacciava il termine WDL di sette ordini di grandezza.
    """
    logits = torch.zeros(1, N_MOVES)
    mask = torch.zeros(1, N_MOVES, dtype=torch.bool)
    mask[0, :30] = True
    masked = masked_policy_logits(logits, mask)
    target = torch.tensor([5])

    loss = masked_cross_entropy(masked, target, mask, label_smoothing=0.05)
    # Con 30 mosse equiprobabili il valore esatto e ln(30) = 3.401.
    assert loss.item() == pytest.approx(3.401, abs=0.01), f"loss anomala: {loss.item()}"


def test_smoothing_e_senza_smoothing_coincidono_su_distribuzione_uniforme():
    """Se tutte le legali hanno lo stesso logit, lo smoothing non cambia nulla."""
    logits = torch.zeros(1, N_MOVES)
    mask = torch.zeros(1, N_MOVES, dtype=torch.bool)
    mask[0, :30] = True
    masked = masked_policy_logits(logits, mask)
    target = torch.tensor([5])

    senza = masked_cross_entropy(masked, target, mask)
    con = masked_cross_entropy(masked, target, mask, label_smoothing=0.05)
    assert senza.item() == pytest.approx(con.item(), abs=1e-4)


def test_lo_smoothing_penalizza_la_sicurezza_eccessiva():
    """Su una predizione molto sicura, lo smoothing deve alzare la loss."""
    logits = torch.full((1, N_MOVES), -10.0)
    logits[0, 5] = 20.0
    mask = torch.zeros(1, N_MOVES, dtype=torch.bool)
    mask[0, :30] = True
    masked = masked_policy_logits(logits, mask)
    target = torch.tensor([5])

    senza = masked_cross_entropy(masked, target, mask)
    con = masked_cross_entropy(masked, target, mask, label_smoothing=0.05)
    assert con.item() > senza.item()


def test_i_tre_termini_della_loss():
    net = tiny_net()
    batch = fake_batch()
    policy, wdl = net(batch["planes"])
    parts = compute_loss(policy, wdl, batch)

    assert parts.total.requires_grad
    assert parts.policy.item() > 0
    assert parts.wdl.item() > 0
    assert parts.eval_coverage == pytest.approx(1.0)


def test_la_loss_regge_un_batch_senza_eval():
    """Il 66,8% dei campioni non ha valutazione Stockfish: non deve rompere nulla."""
    net = tiny_net()
    batch = fake_batch(with_eval=False)
    policy, wdl = net(batch["planes"])
    parts = compute_loss(policy, wdl, batch)

    assert parts.eval_coverage == 0.0
    assert parts.eval_mse.item() == 0.0
    assert torch.isfinite(parts.total)


def test_la_loss_e_finita_e_ragionevole():
    """Sanita numerica: nessun NaN, nessun valore assurdo."""
    net = tiny_net()
    for n_legal in (1, 5, 30, 100):
        batch = fake_batch(n_legal=n_legal)
        policy, wdl = net(batch["planes"])
        parts = compute_loss(policy, wdl, batch)
        assert torch.isfinite(parts.total), f"loss non finita con {n_legal} mosse legali"
        assert parts.total.item() < 100, f"loss {parts.total.item()} con {n_legal} legali"


def test_metriche_di_accuratezza():
    batch = fake_batch(size=8)
    # Logit costruiti per prevedere esattamente il target: top-1 deve dare 100%.
    policy = torch.zeros(8, N_MOVES)
    policy[torch.arange(8), batch["move_index"]] = 10.0
    wdl = torch.zeros(8, 3)
    wdl[torch.arange(8), batch["wdl"]] = 10.0

    metrics = accuracy_metrics(policy, wdl, batch)
    assert metrics["top1"] == pytest.approx(1.0)
    assert metrics["top5"] == pytest.approx(1.0)
    assert metrics["wdl_acc"] == pytest.approx(1.0)


# --------------------------------------------------------------------------------------
# Optimizer e scheduler
# --------------------------------------------------------------------------------------


def test_il_weight_decay_non_tocca_bias_e_batchnorm():
    net = tiny_net()
    optimizer = build_optimizer(net, TrainConfig(weight_decay=0.1))
    decay, no_decay = optimizer.param_groups
    assert decay["weight_decay"] == 0.1
    assert no_decay["weight_decay"] == 0.0
    # Tutti i parametri devono stare in uno dei due gruppi, nessuno perso per strada.
    assert len(decay["params"]) + len(no_decay["params"]) == len(list(net.parameters()))


def test_lo_scheduler_fa_warmup_poi_decay():
    net = tiny_net()
    config = TrainConfig(lr=1e-3, warmup_steps=100)
    optimizer = build_optimizer(net, config)
    scheduler = build_scheduler(optimizer, config, total_steps=1000)

    lrs = []
    for _ in range(1000):
        lrs.append(scheduler.get_last_lr()[0])
        optimizer.step()
        scheduler.step()

    assert lrs[0] < lrs[50] < lrs[99], "il warmup non sale"
    assert lrs[99] == pytest.approx(1e-3, rel=0.05), "il picco non e al lr configurato"
    assert lrs[-1] < lrs[200], "il decay non scende"
    assert lrs[-1] < 1e-5, "il decay non arriva vicino a zero"


# --------------------------------------------------------------------------------------
# Checkpoint — il criterio del Gate 4 sulla ripresa
# --------------------------------------------------------------------------------------


def test_salva_e_ricarica_rende_i_pesi_identici(tmp_path: Path):
    net = tiny_net()
    optimizer = build_optimizer(net, TrainConfig())
    state = TrainingState(epoch=2, global_step=1234, best_val_top1=0.47)

    path = save_checkpoint(tmp_path / "last.pt", model=net, optimizer=optimizer, state=state)
    assert path.exists()

    ricaricata = tiny_net()
    # Prima del caricamento i pesi sono diversi (inizializzazione casuale).
    assert not torch.equal(
        next(iter(net.state_dict().values())), next(iter(ricaricata.state_dict().values()))
    )

    restored, meta = load_checkpoint(path, model=ricaricata)
    for a, b in zip(net.state_dict().values(), ricaricata.state_dict().values(), strict=True):
        assert torch.equal(a, b)
    assert restored.global_step == 1234
    assert restored.epoch == 2
    assert restored.best_val_top1 == pytest.approx(0.47)
    assert meta.schema_version == 1


def test_ricaricare_riprende_anche_optimizer_e_scheduler(tmp_path: Path):
    """Senza lo stato dell'optimizer, AdamW riparte con i momenti azzerati e la loss
    ha un salto visibile alla ripresa."""
    net = tiny_net()
    config = TrainConfig()
    optimizer = build_optimizer(net, config)
    scheduler = build_scheduler(optimizer, config, total_steps=100)

    # Qualche passo, per popolare lo stato interno di AdamW.
    for _ in range(5):
        batch = fake_batch()
        policy, wdl = net(batch["planes"])
        compute_loss(policy, wdl, batch).total.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    path = save_checkpoint(
        tmp_path / "last.pt", model=net, optimizer=optimizer, scheduler=scheduler
    )

    net2 = tiny_net()
    opt2 = build_optimizer(net2, config)
    sched2 = build_scheduler(opt2, config, total_steps=100)
    load_checkpoint(path, model=net2, optimizer=opt2, scheduler=sched2)

    assert sched2.get_last_lr()[0] == pytest.approx(scheduler.get_last_lr()[0])
    assert len(opt2.state_dict()["state"]) == len(optimizer.state_dict()["state"])


def test_il_checkpoint_e_atomico(tmp_path: Path):
    """Dopo il salvataggio non deve restare nessun file temporaneo."""
    net = tiny_net()
    save_checkpoint(tmp_path / "last.pt", model=net)
    assert not list(tmp_path.glob("*.tmp")), "file temporaneo non rimosso"
    assert (tmp_path / "last.pt").exists()


def test_checkpoint_assente_solleva(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "non_esiste.pt", model=tiny_net())


def test_schema_incompatibile_rifiutato(tmp_path: Path):
    """Un checkpoint di uno schema diverso va rifiutato, non letto a caso."""
    net = tiny_net()
    path = save_checkpoint(tmp_path / "last.pt", model=net)

    payload = torch.load(path, weights_only=False)
    payload["meta"]["schema_version"] = 99
    torch.save(payload, path)

    with pytest.raises(ValueError, match="schema"):
        load_checkpoint(path, model=tiny_net())


def test_lo_stato_rng_viene_ripristinato(tmp_path: Path):
    """Serve perche l'ordine dei dati riprenda identico dopo un'interruzione."""
    import random

    net = tiny_net()
    random.seed(42)
    torch.manual_seed(42)
    path = save_checkpoint(tmp_path / "last.pt", model=net)

    atteso = [random.random() for _ in range(3)]

    random.seed(999)  # sporca lo stato
    load_checkpoint(path, model=tiny_net(), restore_rng=True)
    assert [random.random() for _ in range(3)] == atteso


def test_ripresa_con_map_location(tmp_path: Path):
    """Regressione: caricare con `map_location` su un device sposta anche lo stato RNG,
    e `set_rng_state` pretende un ByteTensor su CPU.

    Sintomo: `TypeError: RNG state must be a torch.ByteTensor` alla ripresa di un run
    interrotto — cioe esattamente quando serve che il checkpoint funzioni. Intercettato
    dallo smoke test del training, non dai test unitari, che caricavano tutti su CPU.
    """
    net = tiny_net()
    path = save_checkpoint(tmp_path / "last.pt", model=net)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state, _ = load_checkpoint(path, model=tiny_net(), restore_rng=True, map_location=device)
    assert state.global_step == 0


def test_history_log_append_only(tmp_path: Path):
    log = HistoryLog(tmp_path / "history.jsonl")
    log.append({"kind": "train", "step": 1, "loss": 3.4})
    log.append({"kind": "val", "step": 2000, "top1": 0.31})

    records = log.read()
    assert len(records) == 2
    assert records[0]["loss"] == 3.4
    assert records[1]["top1"] == 0.31
    assert all("time" in r for r in records)


def test_il_commit_hash_finisce_nel_checkpoint(tmp_path: Path):
    """Regola #2: un risultato che non sai riprodurre non e un risultato."""
    path = save_checkpoint(tmp_path / "last.pt", model=tiny_net(), meta=CheckpointMeta())
    _, meta = load_checkpoint(path)
    assert meta.commit and meta.commit != ""
    assert meta.timestamp_utc.endswith("Z")
