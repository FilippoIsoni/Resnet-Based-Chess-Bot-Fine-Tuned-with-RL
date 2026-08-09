"""Verifica un dataset costruito — i criteri del Gate 3 (§3, Stadio 3).

    python scripts/verify_dataset.py --data data/processed
    python scripts/verify_dataset.py --data data/processed --inspect 20

Controlla gli SHARD SCRITTI, non le strutture in memoria di chi li ha prodotti: se il
packing corrompesse i dati, un controllo fatto a monte non se ne accorgerebbe.

Criteri verificati (PIPELINE.md, Gate 3):
  1. 0 game-id condivisi fra split — non verificabile dagli shard, che non portano il
     game_id: si legge dal manifest, ed e il motivo per cui il manifest esiste
  2. 0 FEN condivise fra split, dopo la deduplica
  3. capping aperture rispettato
  4. round-trip storage: rileggendo gli shard le posizioni sono valide e le mosse legali
  5. distribuzione degli esiti sensata (ne 90% patte ne 90% vittorie bianco)
  6. statistiche stampate e registrate

`--inspect N` stampa N campioni estratti a caso con FEN e mossa decodificata, per
l'**ispezione manuale** che il Gate 3 richiede: nessun controllo automatico la
sostituisce, va fatta con gli occhi almeno una volta.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import chess  # noqa: E402

from chessbot.data import SPLITS, iter_shard_boards, read_shard  # noqa: E402
from chessbot.encoding import decode_move  # noqa: E402

_COLOR = sys.stdout.isatty()
GREEN, RED, GREY, RESET = ("\033[32m", "\033[31m", "\033[90m", "\033[0m") if _COLOR else ("",) * 4


def report(name: str, ok: bool, detail: str = "") -> bool:
    mark = f"{GREEN}[PASS]{RESET}" if ok else f"{RED}[FAIL]{RESET}"
    print(f"  {mark} {name}" + (f"  {GREY}{detail}{RESET}" if detail else ""))
    return ok


def shards_of(data_dir: Path, split: str) -> list[Path]:
    d = data_dir / split
    return sorted(d.glob("shard_*.npy")) if d.exists() else []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "processed")
    ap.add_argument("--inspect", type=int, default=0, help="stampa N campioni a caso")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument(
        "--max-fen-check",
        type=int,
        default=2_000_000,
        help="quante FEN per split tenere in memoria per il controllo di sovrapposizione",
    )
    args = ap.parse_args()

    manifest_path = args.data / "manifest.json"
    if not manifest_path.exists():
        print(f"manifest assente: {manifest_path}")
        print("eseguire prima: python scripts/build_dataset.py --input <dump>")
        return 1

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(f"dataset: {args.data}")
    print(f"  da {manifest.get('input')} — commit {manifest.get('commit')}")
    print(
        f"  {manifest.get('games_kept', 0):,} partite tenute su {manifest.get('games_read', 0):,}"
    )
    print()

    ok = True

    # --- 1. Split per partita (criticita #3) ------------------------------------------
    print("Criticita #3 — split per partita")
    per_split_games = manifest.get("games_per_split", {})
    total_games = sum(per_split_games.values())
    ok &= report(
        "split per PARTITA dichiarato nel manifest",
        manifest.get("split", {}).get("by") == "game",
        f"by={manifest.get('split', {}).get('by')}",
    )
    ok &= report(
        "partite ripartite fra gli split",
        total_games == manifest.get("games_kept", -1),
        f"{per_split_games}",
    )

    # --- 2. Lettura degli shard e FEN per split ---------------------------------------
    print()
    print("Round-trip e sovrapposizioni")
    fens: dict[str, set[str]] = {s: set() for s in SPLITS}
    counts: dict[str, int] = {s: 0 for s in SPLITS}
    illegali = 0
    campioni: list[tuple[str, str, str]] = []
    rng = random.Random(args.seed)

    for split in SPLITS:
        paths = shards_of(args.data, split)
        if not paths:
            continue
        for path in paths:
            array = read_shard(path)
            for board, move_index, _wdl, _eval in iter_shard_boards(array):
                counts[split] += 1
                if len(fens[split]) < args.max_fen_check:
                    fens[split].add(board.board_fen() + " " + ("w" if board.turn else "b"))

                # Round-trip: la mossa decodificata deve essere legale nella posizione
                # ricostruita. E il controllo che smaschera un packing sbagliato.
                try:
                    move = decode_move(board, move_index)
                    if move not in board.legal_moves:
                        illegali += 1
                        if illegali <= 3:
                            print(f"    {RED}mossa illegale{RESET}: {move} in {board.fen()}")
                except Exception as e:
                    illegali += 1
                    if illegali <= 3:
                        print(f"    {RED}decodifica fallita{RESET}: {e}")

                if args.inspect and rng.random() < 0.001 and len(campioni) < args.inspect * 4:
                    campioni.append((split, board.fen(), str(decode_move(board, move_index))))

    ok &= report(
        "round-trip storage: tutte le mosse legali",
        illegali == 0,
        f"{sum(counts.values()):,} posizioni, {illegali} illegali",
    )

    shared = 0
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1 :]:
            common = fens[a] & fens[b]
            if common:
                shared += len(common)
    ok &= report("0 FEN condivise fra split", shared == 0, f"{shared} sovrapposizioni")

    # --- 3. Capping -------------------------------------------------------------------
    print()
    print("Capping e distribuzione")
    cap = manifest.get("split", {}).get("max_occurrences_per_fen", 0)
    top = manifest.get("top_fens", [])
    massimo = max((t["count"] for t in top), default=0)
    ok &= report(
        f"nessuna FEN oltre il cap ({cap:,})", massimo <= cap, f"massimo osservato {massimo:,}"
    )

    results = {int(k): v for k, v in manifest.get("results", {}).items()}
    tot = sum(results.values())
    if tot:
        quote = {k: v / tot for k, v in results.items()}
        patte = quote.get(1, 0.0)
        vittorie = quote.get(2, 0.0)
        sensata = patte < 0.9 and vittorie < 0.9 and quote.get(0, 0.0) < 0.9
        ok &= report(
            "distribuzione esiti sensata",
            sensata,
            f"vittorie bianco {vittorie:.1%}, patte {patte:.1%}, sconfitte {quote.get(0, 0):.1%}",
        )

    # --- 4. Statistiche ---------------------------------------------------------------
    print()
    print("Statistiche")
    for split in SPLITS:
        n = counts[split]
        pct = 100 * n / max(sum(counts.values()), 1)
        print(
            f"  {split:5s}: {n:>12,} posizioni ({pct:5.2f}%) in {len(shards_of(args.data, split))} shard"
        )
    print(f"  totale: {sum(counts.values()):,} posizioni")
    print(f"  con [%eval]: {manifest.get('with_eval', 0):,}")
    print(f"  scartate per capping: {manifest.get('dropped_capping', 0):,}")
    print(f"  scartate per deduplica: {manifest.get('dropped_dedup', 0):,}")

    # --- 5. Ispezione manuale ---------------------------------------------------------
    if args.inspect:
        print()
        print(f"Ispezione manuale — {args.inspect} campioni a caso")
        print(f"{GREY}Rigiocarli in python-chess: mossa legale, posizione coerente.{RESET}")
        for split, fen, move in rng.sample(campioni, min(args.inspect, len(campioni))):
            board = chess.Board(fen)
            legale = chess.Move.from_uci(move) in board.legal_moves
            san = board.san(chess.Move.from_uci(move)) if legale else "ILLEGALE"
            print(f"  [{split}] {fen}")
            print(f"        mossa {move} ({san}) — {'ok' if legale else RED + 'ILLEGALE' + RESET}")

    print()
    if ok:
        print(f"  {GREEN}Gate 3: criteri automatici verdi.{RESET}")
        print(f"  {GREY}Resta l'ispezione manuale: --inspect 20{RESET}")
    else:
        print(f"  {RED}Gate 3 ROSSO — correggere prima di procedere (PIPELINE.md §0){RESET}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
