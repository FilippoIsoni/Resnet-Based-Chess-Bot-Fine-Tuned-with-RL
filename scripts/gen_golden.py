"""Genera i file di riferimento in tests/golden/ (Gate 1, §2.5).

    python scripts/gen_golden.py            # rigenera
    python scripts/gen_golden.py --check    # verifica soltanto, non scrive

Il golden e l'OUTPUT atteso dell'encoder sulle posizioni fisse di
`tests/golden/positions_source.py`. Va rigenerato solo quando l'encoding cambia di
proposito — e allora la modifica va motivata in docs/DECISIONS.md, perche invalida
ogni modello gia allenato.

> Questi file devono risultare IDENTICI su Windows e su macOS (Stadio 0-bis di
> PIPELINE.md). Se divergono c'e una dipendenza dalla piattaforma dentro l'encoder:
> ordine di iterazione, dtype di default o endianness. Per questo il tensore viene
> serializzato come lista di interi e non come blob binario, e l'hash e calcolato su
> byte esplicitamente big-endian.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests" / "golden"))

import chess  # noqa: E402
from positions_source import GOLDEN_POSITIONS  # noqa: E402

from chessbot.encoding import (  # noqa: E402
    N_MOVES,
    N_PLANES,
    decode_planes_to_fen,
    encode_board,
    encode_move,
    tensor_digest,
)

GOLDEN_PATH = ROOT / "tests" / "golden" / "positions.json"


def build_entry(name: str, fen: str, uci: str) -> dict:
    board = chess.Board(fen)
    move = chess.Move.from_uci(uci)
    if move not in board.legal_moves:
        raise SystemExit(f"[{name}] mossa {uci} non legale nella posizione {fen}")

    planes = encode_board(board, repetitions=0)
    index = encode_move(board, move)

    # Piani non nulli: rende il diff leggibile quando il golden cambia, e cattura
    # regressioni che l'hash segnalerebbe senza spiegare.
    nonzero = {str(p): round(float(planes[p].sum()), 4) for p in range(N_PLANES) if planes[p].any()}

    return {
        "name": name,
        "fen": fen,
        "move_uci": uci,
        "side_to_move": "w" if board.turn == chess.WHITE else "b",
        "move_index": int(index),
        "tensor_sha256": tensor_digest(planes),
        "plane_sums": nonzero,
        "piece_count": len(board.piece_map()),
        "reconstructed_fen": decode_planes_to_fen(planes, side_to_move=board.turn),
    }


def build_golden() -> dict:
    entries = [build_entry(*p) for p in GOLDEN_POSITIONS]
    indices = [e["move_index"] for e in entries]
    if any(not 0 <= i < N_MOVES for i in indices):
        raise SystemExit("indice fuori da [0, 4671] — encoder rotto")
    return {
        "_comment": (
            "Riferimento del Gate 1 (§2.5). Rigenerare con scripts/gen_golden.py solo "
            "se l'encoding cambia di proposito, motivando in docs/DECISIONS.md. "
            "Deve essere identico su Windows e macOS."
        ),
        "schema_version": 1,
        "n_planes": N_PLANES,
        "n_moves": N_MOVES,
        "positions": entries,
    }


def dumps(data: dict) -> str:
    # sort_keys + newline finale: il file deve essere byte-identico fra macchine.
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="verifica senza scrivere")
    args = ap.parse_args()

    payload = dumps(build_golden())

    if args.check:
        if not GOLDEN_PATH.exists():
            print("golden assente — eseguire: python scripts/gen_golden.py")
            return 1
        current = GOLDEN_PATH.read_text(encoding="utf-8")
        if current != payload:
            print("golden DIVERSO da quello che l'encoder produce ora.")
            print("Se il cambiamento e voluto: rigenerare e motivare in docs/DECISIONS.md.")
            return 1
        print(f"golden invariato ({len(GOLDEN_POSITIONS)} posizioni)")
        return 0

    GOLDEN_PATH.write_text(payload, encoding="utf-8", newline="\n")
    print(f"scritto {GOLDEN_PATH.relative_to(ROOT)} — {len(GOLDEN_POSITIONS)} posizioni")
    return 0


if __name__ == "__main__":
    sys.exit(main())
