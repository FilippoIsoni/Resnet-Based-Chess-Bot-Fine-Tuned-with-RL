"""Stato del motore per l'API e caricamento del checkpoint (Stadio 7).

Il checkpoint RL (`runs/rl/main/state.pt`) ha un formato non-standard: un
dict con chiave `"champion"` che contiene lo state_dict grezzo, prodotto da
`scripts/run_rl.py` e gia consumato cosi da `scripts/run_elo_ladder.py` /
`scripts/measure_rl_gain.py`. Non passa da `training.checkpoint.load_checkpoint`,
che si aspetta lo schema versionato dei checkpoint supervisionati.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

from chessbot.search import Evaluator, SearchConfig

from .rate_limit import InMemoryRateLimiter

# torch NON si importa qui. Vale la stessa ragione documentata in
# `search/mcts.py`: costa 489 MB di RSS, e servendo ONNX non serve affatto.
# Il ripiego su torch lo importa dentro `load_model`, dove viene usato.

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]

DEFAULT_RL_CHECKPOINT = ROOT / "runs" / "rl" / "main" / "state.pt"
DEFAULT_SUPERVISED_CHECKPOINT = ROOT / "runs" / "supervised" / "best.pt"
DEFAULT_ONNX_MODEL = ROOT / "runs" / "onnx" / "chessbot.onnx"


def _hard_simulations() -> int:
    """Leva per abbassare `hard` a 400 su hardware lento, senza toccare codice
    (decisione da prendere sui tempi misurati sullo Space reale, non ora)."""
    return int(os.environ.get("CHESSBOT_HARD_SIMULATIONS", "800"))


LEVEL_SEARCH_CONFIGS: dict[str, SearchConfig] = {
    # temperature=0.8 su easy: campiona fra le mosse ben visitate invece di
    # prendere sempre la piu visitata, per varieta — non e un indebolimento
    # aggiuntivo, quello lo fanno gia le 50 simulazioni (docs/API_CONTRACT.md).
    "easy": SearchConfig(simulations=50, temperature=0.8),
    "medium": SearchConfig(simulations=200),
    "hard": SearchConfig(simulations=_hard_simulations()),
}


@dataclass
class EngineState:
    """Tutto cio di cui le route hanno bisogno, iniettato via dependency
    (vedi `deps.py`) — cosi i test possono sovrascriverlo interamente senza
    toccare `app.state`."""

    evaluator: Evaluator | None
    device: str
    """Sempre "cpu" in produzione. Stringa e non `torch.device` perche questo
    modulo non importa torch (vedi la nota sugli import)."""
    model_loaded: bool
    start_time: float
    rate_limiter: InMemoryRateLimiter
    busy: bool = False
    """Serializza le ricerche MCTS: letto/scritto senza `await` in mezzo,
    quindi atomico nel modello a singolo event loop di asyncio. Presuppone
    un solo worker Uvicorn."""


def _onnx_model_path() -> Path:
    return Path(os.environ.get("CHESSBOT_ONNX_MODEL", str(DEFAULT_ONNX_MODEL)))


def _rl_checkpoint_path() -> Path:
    return Path(os.environ.get("CHESSBOT_RL_CHECKPOINT", str(DEFAULT_RL_CHECKPOINT)))


def _supervised_checkpoint_path() -> Path:
    return Path(
        os.environ.get("CHESSBOT_SUPERVISED_CHECKPOINT", str(DEFAULT_SUPERVISED_CHECKPOINT))
    )


def load_model(device: str | None = None) -> tuple[Evaluator | None, str]:
    """Carica i pesi in ordine di preferenza: ONNX -> RL -> supervisionato -> nessuno.

    Non solleva mai: un checkpoint assente o non caricabile e un motivo per
    restituire `evaluator=None`, non per far fallire l'avvio del processo.
    `/health` deve poter rispondere `model_loaded: false` invece di crashare
    (entrambi i checkpoint sono gitignored e assenti in un checkout pulito).

    **ONNX per primo, quando c'e.** Il servizio con torch occupa 608 MB di RSS,
    contro i 100 della stessa rete servita da onnxruntime: quasi tutti i piani
    di hosting gratuiti si fermano a 512 MB, quindi la differenza decide se il
    backend sta online oppure no. La parita numerica fra i due percorsi e
    verificata da `tests/unit/test_onnx_parity.py`.

    Il ripiego su torch resta perche in sviluppo il `.onnx` di solito non c'e
    (si rigenera da un checkpoint, entrambi gitignored) e perche il tipo di
    ritorno annotato `Evaluator` copre entrambi: `OnnxEvaluator` espone la
    stessa `evaluate()` e l'MCTS non distingue i due casi.
    """
    resolved_device = device or os.environ.get("CHESSBOT_DEVICE", "cpu")

    onnx_path = _onnx_model_path()
    if onnx_path.exists():
        try:
            from .onnx_evaluator import OnnxEvaluator

            logger.info("motore servito da ONNX: %s", onnx_path)
            # `type: ignore`: OnnxEvaluator non eredita da Evaluator (non deve
            # importare torch), ma ne implementa l'unico metodo che la ricerca
            # usa. La compatibilita e verificata dai test di parita.
            return OnnxEvaluator(onnx_path), resolved_device  # type: ignore[return-value]
        except Exception:
            logger.exception("modello ONNX presente ma non caricabile: %s", onnx_path)

    # La rete si costruisce solo se c'e davvero un checkpoint da caricarci
    # dentro: nel caso comune di sviluppo (nessun checkpoint su disco, entrambi
    # gitignored) questo rende load_model() pressoche istantaneo, che e anche
    # cio che tiene i test `fast` veloci senza bisogno di un caso speciale.
    rl_path = _rl_checkpoint_path()
    sup_path = _supervised_checkpoint_path()

    if rl_path.exists() or sup_path.exists():
        # Import qui e non in cima: si paga solo quando si serve davvero con
        # torch, cioe in sviluppo o se manca il modello ONNX.
        import torch

        from chessbot.model.network import ChessNet, NetworkConfig
        from chessbot.training.checkpoint import load_checkpoint

        torch_device = torch.device(resolved_device)

        if rl_path.exists():
            try:
                net = ChessNet(NetworkConfig()).to(torch_device)
                payload = torch.load(rl_path, map_location=torch_device, weights_only=False)
                net.load_state_dict(payload["champion"])
                return Evaluator(net, torch_device), resolved_device
            except Exception:
                logger.exception("checkpoint RL presente ma non caricabile: %s", rl_path)

        if sup_path.exists():
            try:
                net = ChessNet(NetworkConfig()).to(torch_device)
                load_checkpoint(sup_path, model=net, restore_rng=False, map_location=torch_device)
                return Evaluator(net, torch_device), resolved_device
            except Exception:
                logger.exception(
                    "checkpoint supervisionato presente ma non caricabile: %s", sup_path
                )

    logger.warning(
        "nessun checkpoint disponibile (%s, %s) — model_loaded restera False", rl_path, sup_path
    )
    return None, resolved_device


def new_start_time() -> float:
    """Isolato in una funzione per poterlo mockare facilmente nei test di uptime."""
    return time.monotonic()


__all__ = ["LEVEL_SEARCH_CONFIGS", "EngineState", "load_model", "new_start_time"]
