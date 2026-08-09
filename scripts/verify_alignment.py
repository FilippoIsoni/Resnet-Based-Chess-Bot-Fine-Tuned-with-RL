"""Verifica l'allineamento posizione <-> mossa su tutto il dataset (Gate 3).

    python scripts/verify_alignment.py --data data/processed
    python scripts/verify_alignment.py --data data/processed --split val
    python scripts/verify_alignment.py --data data/processed --limit 100000
    python scripts/verify_alignment.py --data data/processed --dump-failures out.jsonl

Sola lettura: non modifica ne rigenera nulla.

Verdetto quantitativo, non ispezione visiva. Per ogni campione classifica l'esito in
una categoria, cosi un eventuale disallineamento dice anche di che tipo e:

| Categoria | Significato |
|---|---|
| `ok` | mossa legale nella posizione, decodificata con l'orientamento del dataset |
| `ok_mirror` | illegale cosi, legale specchiando le due case |
| `ok_next_ply` | illegale cosi, legale nella posizione successiva dello shard |
| `ok_prev_ply` | illegale cosi, legale nella posizione precedente dello shard |
| `illegal` | illegale in tutti i casi sopra |

> **Nota sul mirroring — leggere prima di interpretare `ok_mirror`.**
> In questo dataset l'indice mossa e salvato **gia orientato dal lato che muove**
> (`storage.encode_samples` usa `encoding.encode_move`, non `move_to_index`). Non e un
> difetto: e la contromisura alla criticita #1, applicata una volta in preprocessing
> invece che ad ogni epoca. La decodifica corretta e quindi `decode_move`, che riapplica
> il flip — ed e cio che questo script usa per la categoria `ok`.
>
> Di conseguenza `ok_mirror` **non** significa "mosse in coordinate canoniche": quello e
> gia il caso, ed e gestito. Un `ok_mirror` diffuso significherebbe che il flip viene
> applicato **due volte** o **zero volte**, cioe un bug vero.

> **Limite noto.** Gli shard non conservano il `game_id`: e escluso dallo storage
> perche costerebbe piu di tutti gli altri campi messi insieme (§2.6), e serve solo per
> lo split, che avviene a monte. Le categorie `next_ply`/`prev_ply` confrontano quindi
> campioni **adiacenti nello shard**, che quasi sempre appartengono alla stessa partita
> ma al confine fra due partite no. L'effetto e un tasso di falsi `next_ply`/`prev_ply`
> dell'ordine di 1/73 (una posizione per partita), che va tenuto presente solo se quelle
> categorie risultano non nulle.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import chess  # noqa: E402

from chessbot.data import SPLITS, read_shard, unpack_board  # noqa: E402
from chessbot.encoding import decode_move  # noqa: E402

_COLOR = sys.stdout.isatty()
GREEN, RED, YELLOW, GREY, BOLD, RESET = (
    ("\033[32m", "\033[31m", "\033[33m", "\033[90m", "\033[1m", "\033[0m")
    if _COLOR
    else ("", "", "", "", "", "")
)

CATEGORIES = ("ok", "ok_mirror", "ok_next_ply", "ok_prev_ply", "illegal")

# Soglia sotto la quale lo script esce con codice != 0, per l'uso come gate in CI.
OK_THRESHOLD = 0.99


def classify(
    board: chess.Board,
    move_index: int,
    prev: chess.Board | None,
    nxt: chess.Board | None,
) -> tuple[str, chess.Move | None]:
    """Classifica una coppia (posizione, indice mossa).

    Returns:
        `(categoria, mossa decodificata)`. La mossa e None se l'indice non e nemmeno
        decodificabile — caso che finisce in `illegal`.
    """
    try:
        move = decode_move(board, move_index)
    except Exception:
        return "illegal", None

    if move in board.legal_moves:
        return "ok", move

    mirrored = chess.Move(
        chess.square_mirror(move.from_square),
        chess.square_mirror(move.to_square),
        promotion=move.promotion,
    )
    if mirrored in board.legal_moves:
        return "ok_mirror", move

    if nxt is not None and move in nxt.legal_moves:
        return "ok_next_ply", move
    if prev is not None and move in prev.legal_moves:
        return "ok_prev_ply", move

    return "illegal", move


def turn_matches_piece(board: chess.Board, move: chess.Move) -> bool:
    """Il pezzo mosso appartiene al lato al tratto?

    Controllo separato da `classify`: un disallineamento del solo flag del tratto e un
    bug distinto dalla mossa illegale, e con `decode_move` non puo comunque presentarsi
    su una mossa legale — ma verificarlo costa nulla e rende il verdetto non ambiguo.
    """
    piece = board.piece_at(move.from_square)
    return piece is not None and piece.color == board.turn


def shards_of(data_dir: Path, split: str) -> list[Path]:
    d = data_dir / split
    return sorted(d.glob("shard_*.npy")) if d.exists() else []


def verify_split(
    data_dir: Path,
    split: str,
    *,
    limit: int | None,
    failures: list[dict],
    max_failures: int,
) -> tuple[Counter[str], int]:
    """Scansiona uno split. Restituisce (conteggi per categoria, disallineamenti di tratto)."""
    counts: Counter[str] = Counter()
    turn_mismatch = 0
    seen = 0
    started = time.perf_counter()
    last_report = 0.0

    for path in shards_of(data_dir, split):
        array = read_shard(path)

        boards: list[chess.Board] = []
        indices: list[int] = []
        for row in array:
            boards.append(unpack_board(row["bitboards"], int(row["state"]), int(row["clock"])))
            indices.append(int(row["move_index"]))

        for i, (board, move_index) in enumerate(zip(boards, indices, strict=True)):
            prev = boards[i - 1] if i > 0 else None
            nxt = boards[i + 1] if i + 1 < len(boards) else None

            category, move = classify(board, move_index, prev, nxt)
            counts[category] += 1

            if category == "ok" and move is not None and not turn_matches_piece(board, move):
                turn_mismatch += 1

            if category != "ok" and len(failures) < max_failures:
                failures.append(
                    {
                        "split": split,
                        "shard": path.name,
                        "index": i,
                        "fen": board.fen(),
                        "move_uci": str(move) if move else None,
                        "move_index": move_index,
                        "category": category,
                    }
                )

            seen += 1
            if limit is not None and seen >= limit:
                return counts, turn_mismatch

            now = time.perf_counter()
            if now - last_report >= 15:
                rate = seen / (now - started)
                pct = 100 * counts["ok"] / seen
                print(
                    f"  [{split}] {seen:,} campioni, ok {pct:.3f}% — {rate:.0f}/s",
                    flush=True,
                )
                last_report = now

    return counts, turn_mismatch


def diagnose(counts: Counter[str], total: int, turn_mismatch: int) -> list[str]:
    """Traduce i conteggi in una diagnosi e un'azione, come da tabella del Gate 3."""
    if not total:
        return ["nessun campione letto — dataset assente o --limit 0"]

    frac = {c: counts[c] / total for c in CATEGORIES}
    lines: list[str] = []

    if frac["ok"] >= 0.9999:
        lines.append(f"{GREEN}Dataset allineato.{RESET} Posizione e mossa corrispondono.")
        lines.append(
            "Se un tool di ispezione mostrava un disallineamento, il bug e nella sua"
            " stampa, non nei dati."
        )
    elif frac["ok_mirror"] > 0.5:
        lines.append(f"{RED}Il flip dell'orientamento e applicato male.{RESET}")
        lines.append(
            "Le mosse risultano legali solo specchiate, ma il dataset le salva GIA"
            " orientate: il flip viene applicato due volte, o zero. Correggere"
            " `storage.encode_samples` o `encoding.decode_move` — NON i dati."
        )
    elif frac["ok_next_ply"] + frac["ok_prev_ply"] > 0.05:
        lines.append(f"{RED}Off-by-one reale nel preprocessing.{RESET}")
        lines.append(
            "Le mosse appartengono alla posizione adiacente: `samples_from_game` associa"
            " la mossa alla board sbagliata. Dataset da rigenerare dopo il fix."
        )
    elif 0.4 < frac["ok"] < 0.6:
        lines.append(f"{RED}Sospetto off-by-one.{RESET}")
        lines.append(
            "Circa meta dei campioni resta legale per caso: e la firma di uno sfasamento"
            " di un ply. Dataset da rigenerare."
        )
    elif frac["illegal"] > 0:
        lines.append(f"{YELLOW}Casi speciali gestiti male nel parsing.{RESET}")
        lines.append(
            f"{counts['illegal']:,} campioni illegali ({100 * frac['illegal']:.4f}%)."
            " Ispezionare i falliti con --dump-failures: i sospetti sono arrocco,"
            " en passant e promozione."
        )
    else:
        lines.append(f"{YELLOW}Nessuna categoria dominante.{RESET} Ispezionare i falliti.")

    if turn_mismatch:
        lines.append(
            f"{RED}{turn_mismatch:,} campioni muovono un pezzo del colore sbagliato.{RESET}"
            " Il flag del tratto non corrisponde alla mossa: bug distinto dalla legalita."
        )

    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "processed")
    ap.add_argument("--split", choices=SPLITS, default=None, help="default: tutti")
    ap.add_argument("--limit", type=int, default=None, help="campioni per split")
    ap.add_argument("--dump-failures", type=Path, default=None)
    ap.add_argument("--max-failures", type=int, default=200)
    args = ap.parse_args()

    if not (args.data / "manifest.json").exists():
        print(f"dataset assente: {args.data}")
        return 1

    splits = [args.split] if args.split else list(SPLITS)
    failures: list[dict] = []
    totals: Counter[str] = Counter()
    total_turn_mismatch = 0
    per_split: dict[str, Counter[str]] = {}

    print(f"{BOLD}Verifica allineamento posizione <-> mossa{RESET}")
    print(f"  dataset: {args.data}")
    print(f"  split  : {', '.join(splits)}")
    if args.limit:
        print(f"  limite : {args.limit:,} campioni per split")
    print()

    started = time.perf_counter()
    for split in splits:
        counts, mismatch = verify_split(
            args.data,
            split,
            limit=args.limit,
            failures=failures,
            max_failures=args.max_failures,
        )
        per_split[split] = counts
        totals.update(counts)
        total_turn_mismatch += mismatch
    elapsed = time.perf_counter() - started

    total = sum(totals.values())

    print()
    print(f"{BOLD}Risultati per split{RESET}")
    header = f"  {'split':6s} {'campioni':>12s}  " + "  ".join(f"{c:>12s}" for c in CATEGORIES)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for split in splits:
        c = per_split[split]
        n = sum(c.values())
        row = f"  {split:6s} {n:>12,}  " + "  ".join(f"{c[cat]:>12,}" for cat in CATEGORIES)
        print(row)

    print()
    print(f"{BOLD}Totale{RESET}")
    for cat in CATEGORIES:
        n = totals[cat]
        pct = 100 * n / total if total else 0.0
        color = GREEN if (cat == "ok" and pct > 99) else (RED if n and cat != "ok" else GREY)
        print(f"  {color}{cat:14s}{RESET} {n:>12,}  {pct:7.4f}%")
    print(f"  {'campioni':14s} {total:>12,}")
    print(f"  {'tratto errato':14s} {total_turn_mismatch:>12,}")
    print(f"  {GREY}{elapsed / 60:.1f} min, {total / max(elapsed, 1e-9):.0f} campioni/s{RESET}")

    print()
    print(f"{BOLD}Diagnosi{RESET}")
    for line in diagnose(totals, total, total_turn_mismatch):
        print(f"  {line}")

    if args.dump_failures and failures:
        args.dump_failures.parent.mkdir(parents=True, exist_ok=True)
        with open(args.dump_failures, "w", encoding="utf-8", newline="\n") as fh:
            for f in failures:
                fh.write(json.dumps(f, ensure_ascii=False) + "\n")
        print()
        print(f"  {len(failures)} falliti salvati in {args.dump_failures}")

    ok_rate = totals["ok"] / total if total else 0.0
    if ok_rate < OK_THRESHOLD or total_turn_mismatch:
        print()
        print(
            f"  {RED}ROSSO — ok {100 * ok_rate:.4f}% sotto la soglia del {100 * OK_THRESHOLD:.0f}%{RESET}"
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
