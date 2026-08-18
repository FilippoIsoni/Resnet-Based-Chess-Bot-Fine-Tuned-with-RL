"""Valutatore ONNX: la stessa interfaccia di `search.Evaluator`, senza torch.

## Perche esiste

Il backend con torch occupa **608 MB** di RSS a regime, di cui 489 solo per
`import torch`. Praticamente tutti i piani di hosting gratuiti si fermano a
512 MB, quindi il servizio non parte proprio — va in OOM prima ancora di
caricare i pesi. Con onnxruntime la stessa rete ne occupa **100 MB**: sei volte
meno, e il problema di hosting sparisce.

E la strada che l'Appendice B del piano prescriveva fin dall'inizio. Era stata
scartata (docs/DECISIONS.md, 2026-08-17) perche su Hugging Face Spaces il peso
non era un vincolo — poi HF ha spostato gli Space Docker su piano a pagamento e
quella premessa e caduta.

## Sostituibilita

`Evaluator` e documentato come sostituibile e la ricerca lo tratta come tale:
`run_mcts` chiama solo `evaluate(boards)`. Questa classe replica quel metodo
firma per firma, quindi l'MCTS non distingue le due implementazioni e non
cambia una riga.

Il calcolo qui dentro **ricalca deliberatamente** quello di
`search/mcts.py::Evaluator.evaluate`: stessa mascheratura delle mosse illegali
(criticita #12), stessa normalizzazione dei priori, stesso WDL -> scalare. La
duplicazione e voluta: unificarli richiederebbe che il motore importi
onnxruntime, e il senso di questo file e proprio non avere quella dipendenza
nel percorso di training.

La parita numerica fra le due e verificata da `tests/unit/test_onnx_parity.py`,
che confronta gli output posizione per posizione. La soglia del piano e 1e-3;
gli scarti misurati stanno intorno a 1e-05.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chess
import numpy as np

from chessbot.encoding import encode_board, encode_move

# Stesso valore usato da `model.network.masked_policy_logits` per fp32: un
# numero abbastanza negativo da azzerare la softmax, ma non -inf, che
# produrrebbe NaN se una riga non avesse mosse legali.
MASK_FILL = np.finfo(np.float32).min / 2


class OnnxEvaluator:
    """Valuta posizioni con onnxruntime invece che con torch.

    Interfaccia identica a `chessbot.search.Evaluator`: `evaluate(boards)` ->
    lista di `(priori sulle mosse legali, valore in [-1, 1])`, col valore dal
    punto di vista di chi muove.
    """

    def __init__(self, model_path: str | Path, *, intra_op_threads: int = 2) -> None:
        # Import locale: chi non usa il backend ONNX non deve pagare
        # l'import, e onnxruntime non e fra le dipendenze di training.
        import onnxruntime as ort

        options = ort.SessionOptions()
        # Su un container gratuito i core sono pochi e condivisi: lasciare che
        # onnxruntime ne apra quanti ne vede porta a oversubscription e batch
        # piu lenti, non piu veloci.
        options.intra_op_num_threads = intra_op_threads
        options.inter_op_num_threads = 1

        self.session = ort.InferenceSession(
            str(model_path), options, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def evaluate(self, boards: list[chess.Board]) -> list[tuple[dict[chess.Move, float], float]]:
        """Da posizioni a (priori sulle mosse legali, valore).

        Ricalca `search/mcts.py::Evaluator.evaluate`; vedi la docstring del
        modulo per il perche della duplicazione.
        """
        if not boards:
            return []

        legal_moves = [list(b.legal_moves) for b in boards]
        indices = [
            [encode_move(b, m) for m in moves] for b, moves in zip(boards, legal_moves, strict=True)
        ]

        planes = np.stack([encode_board(b, repetitions=0) for b in boards]).astype(np.float32)
        policy_logits, wdl_logits = self.session.run(None, {self.input_name: planes})

        # Mascheratura identica a quella del training (criticita #12): le mosse
        # illegali vanno azzerate PRIMA della softmax, non dopo.
        masked = np.full_like(policy_logits, MASK_FILL, dtype=np.float32)
        for row, idx in enumerate(indices):
            if idx:
                masked[row, idx] = policy_logits[row, idx]

        probs = _softmax(masked)
        wdl = _softmax(wdl_logits.astype(np.float32))
        values = wdl[:, 2] - wdl[:, 0]  # P(vittoria) - P(sconfitta), in [-1, 1]

        out: list[tuple[dict[chess.Move, float], float]] = []
        for i, moves in enumerate(legal_moves):
            selected = probs[i, indices[i]] if indices[i] else np.empty(0, dtype=np.float32)
            total = float(selected.sum())
            if total > 0:
                priors = {m: float(p) / total for m, p in zip(moves, selected, strict=True)}
            else:
                # Tutti i logit schiacciati: ripiega su uniforme invece di
                # dividere per zero. Stesso comportamento dell'Evaluator torch.
                priors = dict.fromkeys(moves, 1.0 / max(len(moves), 1))

            # `encode_board` orienta sempre dal lato che muove, quindi il
            # valore e gia nella prospettiva giusta. Nessuna conversione.
            out.append((priors, float(values[i])))
        return out


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Softmax per righe, stabile numericamente."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def export_to_onnx(checkpoint: Path, output: Path, *, opset: int = 17) -> Path:
    """Converte un checkpoint in un modello ONNX.

    Vive qui e non in uno script perche il test di parita lo richiama: un
    export verificato solo a mano e un export che diverge in silenzio alla
    prossima modifica della rete.

    Nota sui file prodotti: per un modello di questa taglia l'esportatore
    scrive i pesi in un `.data` affiancato, quindi vanno spostati insieme.
    """
    import torch  # solo in export: il servizio in produzione non importa torch

    from chessbot.model.network import ChessNet, NetworkConfig
    from chessbot.training.checkpoint import load_checkpoint

    net = ChessNet(NetworkConfig())
    payload: Any = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "champion" in payload:
        net.load_state_dict(payload["champion"])  # formato RL
    else:
        load_checkpoint(checkpoint, model=net, restore_rng=False, map_location="cpu")
    net.eval()

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        net,
        (torch.zeros(1, 19, 8, 8),),
        str(output),
        input_names=["planes"],
        output_names=["policy", "wdl"],
        # Il batch deve restare dinamico: l'MCTS valuta da 1 a `batch_size_leaves`
        # posizioni per chiamata, e un modello a batch fisso fallirebbe su tutte
        # le dimensioni tranne una.
        dynamic_shapes={"planes": {0: torch.export.Dim("batch")}},
        opset_version=opset,
    )
    return output


__all__ = ["OnnxEvaluator", "export_to_onnx"]
