"""Converte la rete in ONNX per il deployment (Appendice B).

    python scripts/export_onnx.py
    python scripts/export_onnx.py --checkpoint runs/supervised/best.pt

## Perche

Il backend con torch occupa **608 MB** di RSS, di cui 489 solo per
`import torch`. Praticamente ogni piano di hosting gratuito si ferma a 512 MB,
quindi il servizio non parte: va in OOM prima ancora di caricare i pesi. La
stessa rete servita da onnxruntime ne occupa **100 MB**.

Era la scelta originale del piano. L'avevamo scartata quando Hugging Face
offriva Space Docker gratuiti e il peso non contava; da luglio 2026 quegli
Space richiedono un piano a pagamento, e la premessa e caduta.

## Cosa produce

Due file, che vanno spostati **insieme**: un modello di questa taglia tiene i
pesi in un `.data` affiancato, e il `.onnx` da solo e un guscio da pochi
kilobyte.

    runs/onnx/chessbot.onnx        struttura della rete
    runs/onnx/chessbot.onnx.data   i pesi, ~46 MB

Sono artefatti di build, non file da versionare: `.gitignore` esclude `*.onnx`
e `scripts/check.py` lo elenca fra le estensioni vietate.

## La verifica

Non basta che l'export non dia errore: una rete che risponde *quasi* come
quella addestrata non crasha, gioca solo peggio. Questo script confronta i due
percorsi su posizioni diverse e rifiuta il risultato se divergono oltre 1e-3,
la soglia fissata dal piano. La stessa verifica gira come test in
`tests/unit/test_onnx_parity.py`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import chess  # noqa: E402
import torch  # noqa: E402

from chessbot.api.onnx_evaluator import OnnxEvaluator, export_to_onnx  # noqa: E402
from chessbot.model.network import ChessNet, NetworkConfig  # noqa: E402
from chessbot.search import Evaluator  # noqa: E402
from chessbot.training.checkpoint import load_checkpoint  # noqa: E402

GREEN, RED, GREY, BOLD, RESET = (
    ("\033[32m", "\033[31m", "\033[90m", "\033[1m", "\033[0m")
    if sys.stdout.isatty()
    else ("", "", "", "", "")
)

TOLLERANZA = 1e-3

# Apertura, mediogioco col nero al tratto (orientamento invertito, criticita
# #1), finale spoglio, posizione con arrocchi, re quasi mattato.
POSIZIONI = [
    chess.STARTING_FEN,
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    "8/5k2/8/8/3K4/8/5Q2/8 w - - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/8/8/8/8/5k2/6q1/7K w - - 0 1",
]


def carica_torch(checkpoint: Path) -> ChessNet:
    net = ChessNet(NetworkConfig())
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "champion" in payload:
        net.load_state_dict(payload["champion"])
    else:
        load_checkpoint(checkpoint, model=net, restore_rng=False, map_location="cpu")
    net.eval()
    return net


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "runs" / "rl" / "main" / "state.pt")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "onnx" / "chessbot.onnx")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    if not args.checkpoint.exists():
        print(f"{RED}checkpoint assente: {args.checkpoint}{RESET}")
        return 1

    print(f"{BOLD}Export ONNX{RESET}")
    print(f"  da:  {args.checkpoint.relative_to(ROOT)}")
    print(f"  a:   {args.out.relative_to(ROOT)}")
    print()

    export_to_onnx(args.checkpoint, args.out, opset=args.opset)

    peso = sum(f.stat().st_size for f in args.out.parent.glob(f"{args.out.name}*") if f.is_file())
    print(f"  esportato, {peso / 1024 / 1024:.1f} MB in totale")
    print()

    # --- verifica di parita ---------------------------------------------------------
    print(f"{BOLD}Parita con PyTorch{RESET}  {GREY}(soglia {TOLLERANZA:g}){RESET}")

    torch_eval = Evaluator(carica_torch(args.checkpoint), torch.device("cpu"))
    onnx_eval = OnnxEvaluator(args.out)
    boards = [chess.Board(f) for f in POSIZIONI]

    peggior_priore = 0.0
    peggior_valore = 0.0
    mosse_diverse = 0

    for fen, (pt, vt), (po, vo) in zip(
        POSIZIONI, torch_eval.evaluate(boards), onnx_eval.evaluate(boards), strict=True
    ):
        if pt.keys() != po.keys():
            print(f"  {RED}mosse legali diverse su {fen}{RESET}")
            return 1

        dp = max((abs(pt[m] - po[m]) for m in pt), default=0.0)
        dv = abs(vt - vo)
        peggior_priore = max(peggior_priore, dp)
        peggior_valore = max(peggior_valore, dv)

        if pt:
            stessa = max(pt, key=lambda m: pt[m]) == max(po, key=lambda m: po[m])
            mosse_diverse += 0 if stessa else 1
            marchio = f"{GREEN}=={RESET}" if stessa else f"{RED}!={RESET}"
        else:
            marchio = f"{GREY}--{RESET}"

        print(f"  {marchio} {fen[:38]:40s} priori {dp:.1e}  valore {dv:.1e}")

    print()
    scarto = max(peggior_priore, peggior_valore)
    ok = scarto < TOLLERANZA and mosse_diverse == 0

    print(f"  scarto massimo: {scarto:.2e}")
    print(f"  mossa preferita diversa in {mosse_diverse} posizioni su {len(POSIZIONI)}")
    print()

    if ok:
        print(f"  {GREEN}Il modello ONNX e utilizzabile.{RESET}")
        print(f"  Il backend lo usera in automatico se trova {args.out.name} in runs/onnx/.")
        return 0

    print(f"  {RED}Divergenza oltre la soglia: non usare questo modello.{RESET}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
