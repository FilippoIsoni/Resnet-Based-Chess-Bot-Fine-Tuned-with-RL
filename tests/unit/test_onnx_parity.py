"""Il modello ONNX risponde come quello PyTorch.

E la verifica che l'Appendice B chiede esplicitamente ("riutilizzare la
batteria di §2.5, output identici entro 1e-3") e senza la quale servire ONNX e
un salto nel buio: una rete che risponde *quasi* come quella addestrata non
crasha, gioca solo un po' peggio — la stessa categoria di bug silenzioso di
tutte le criticita di questo progetto.

I test sono marcati `slow` perche esportano il modello vero, e vengono saltati
se il checkpoint non c'e (e gitignored, quindi assente in un checkout pulito).
"""

from __future__ import annotations

from pathlib import Path

import chess
import numpy as np
import pytest
import torch

from chessbot.api.onnx_evaluator import OnnxEvaluator, export_to_onnx
from chessbot.encoding import encode_board
from chessbot.model.network import ChessNet, NetworkConfig
from chessbot.search import Evaluator

pytestmark = pytest.mark.slow

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "runs" / "rl" / "main" / "state.pt"

# La soglia del piano. Gli scarti misurati stanno intorno a 1e-05, tre ordini
# di grandezza sotto: se un giorno questo test fallisce di poco non e rumore
# numerico, e un cambiamento vero nella rete o nell'esportatore.
TOLLERANZA = 1e-3

# Posizioni scelte per coprire casi diversi: apertura, mediogioco con il nero
# al tratto (cambia l'orientamento della codifica, criticita #1), un finale
# spoglio, una posizione con arrocchi disponibili da entrambe le parti.
POSIZIONI = [
    chess.STARTING_FEN,
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    "8/5k2/8/8/3K4/8/5Q2/8 w - - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/8/8/8/8/5k2/6q1/7K w - - 0 1",  # bianco quasi mattato
]


@pytest.fixture(scope="module")
def modello_onnx(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if not CHECKPOINT.exists():
        pytest.skip(f"checkpoint assente: {CHECKPOINT}")
    destinazione = tmp_path_factory.mktemp("onnx") / "chessbot.onnx"
    return export_to_onnx(CHECKPOINT, destinazione)


@pytest.fixture(scope="module")
def rete_torch() -> ChessNet:
    if not CHECKPOINT.exists():
        pytest.skip(f"checkpoint assente: {CHECKPOINT}")
    net = ChessNet(NetworkConfig())
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    net.load_state_dict(payload["champion"])
    net.eval()
    return net


def test_logit_grezzi_coincidono(modello_onnx: Path, rete_torch: ChessNet) -> None:
    """Confronto diretto delle uscite della rete, prima di ogni elaborazione.

    E il test piu stringente: qualsiasi divergenza introdotta dall'export si
    vede qui, non filtrata da softmax o normalizzazioni che potrebbero
    mascherarla.
    """
    import onnxruntime as ort

    sessione = ort.InferenceSession(str(modello_onnx), providers=["CPUExecutionProvider"])

    for fen in POSIZIONI:
        board = chess.Board(fen)
        planes = np.stack([encode_board(board, repetitions=0)]).astype(np.float32)

        with torch.no_grad():
            policy_torch, wdl_torch = rete_torch(torch.from_numpy(planes))
        policy_onnx, wdl_onnx = sessione.run(None, {"planes": planes})

        scarto_policy = float(np.abs(policy_torch.numpy() - policy_onnx).max())
        scarto_wdl = float(np.abs(wdl_torch.numpy() - wdl_onnx).max())

        assert scarto_policy < TOLLERANZA, f"policy diverge di {scarto_policy:.2e} su {fen}"
        assert scarto_wdl < TOLLERANZA, f"wdl diverge di {scarto_wdl:.2e} su {fen}"


def test_priori_e_valore_coincidono(modello_onnx: Path, rete_torch: ChessNet) -> None:
    """Confronto a livello di `evaluate()`, cioe di cio che l'MCTS riceve davvero.

    Copre anche la mascheratura e la normalizzazione, che sono reimplementate
    in `OnnxEvaluator` invece che condivise: un errore li non si vedrebbe nel
    test sui logit grezzi.
    """
    torch_eval = Evaluator(rete_torch, torch.device("cpu"))
    onnx_eval = OnnxEvaluator(modello_onnx)

    boards = [chess.Board(fen) for fen in POSIZIONI]
    attesi = torch_eval.evaluate(boards)
    ottenuti = onnx_eval.evaluate(boards)

    assert len(attesi) == len(ottenuti)

    for fen, (priori_t, valore_t), (priori_o, valore_o) in zip(
        POSIZIONI, attesi, ottenuti, strict=True
    ):
        assert priori_t.keys() == priori_o.keys(), f"mosse diverse su {fen}"
        assert abs(valore_t - valore_o) < TOLLERANZA, f"valore diverge su {fen}"

        for mossa in priori_t:
            scarto = abs(priori_t[mossa] - priori_o[mossa])
            assert scarto < TOLLERANZA, f"priore di {mossa.uci()} diverge di {scarto:.2e} su {fen}"


def test_la_mossa_preferita_e_la_stessa(modello_onnx: Path, rete_torch: ChessNet) -> None:
    """Il controllo che conta per la forza di gioco.

    Gli scarti numerici sono irrilevanti finche non cambiano una decisione. Se
    la mossa con il priore piu alto e la stessa, la policy si comporta allo
    stesso modo dove conta.
    """
    torch_eval = Evaluator(rete_torch, torch.device("cpu"))
    onnx_eval = OnnxEvaluator(modello_onnx)
    boards = [chess.Board(fen) for fen in POSIZIONI]

    for fen, (priori_t, _), (priori_o, _) in zip(
        POSIZIONI,
        torch_eval.evaluate(boards),
        onnx_eval.evaluate(boards),
        strict=True,
    ):
        if not priori_t:
            continue
        migliore_t = max(priori_t, key=lambda m: priori_t[m])
        migliore_o = max(priori_o, key=lambda m: priori_o[m])
        assert migliore_t == migliore_o, (
            f"mossa preferita diversa su {fen}: torch {migliore_t.uci()}, onnx {migliore_o.uci()}"
        )


def test_batch_di_dimensione_variabile(modello_onnx: Path) -> None:
    """L'MCTS valuta da 1 a `batch_size_leaves` posizioni per chiamata.

    Un modello esportato a batch fisso fallirebbe su tutte le dimensioni tranne
    quella di export — e siccome la ricerca chiama con batch irregolari, il
    guasto arriverebbe alla prima mossa vera, non al primo test.
    """
    onnx_eval = OnnxEvaluator(modello_onnx)

    for n in (1, 2, 7, 24):
        boards = [chess.Board(POSIZIONI[i % len(POSIZIONI)]) for i in range(n)]
        risultati = onnx_eval.evaluate(boards)
        assert len(risultati) == n
        for priori, valore in risultati:
            assert -1.0 <= valore <= 1.0
            if priori:
                assert abs(sum(priori.values()) - 1.0) < 1e-5


def test_lista_vuota_non_esplode(modello_onnx: Path) -> None:
    assert OnnxEvaluator(modello_onnx).evaluate([]) == []
