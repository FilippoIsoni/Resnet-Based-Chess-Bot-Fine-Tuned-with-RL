"""Scarica un dump PGN di Lichess in data/raw/ (Stadio 3, §2.1).

    python scripts/download_pgn.py --month 2026-07
    python scripts/download_pgn.py --month 2026-07 --check    # solo dimensione, non scarica

I dump stanno su https://database.lichess.org come `.pgn.zst`. Si scaricano compressi e
si leggono in streaming: decomprimerli su disco costerebbe centinaia di GB per nulla.

Il download **riprende** se interrotto (richiesta Range sui byte gia presenti): un dump
da 27 GB su una connessione domestica non e un'operazione che si vuole rifare da capo.
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST_DIR = ROOT / "data" / "raw"
BASE_URL = "https://database.lichess.org/standard"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CHUNK = 1 << 20  # 1 MiB


def url_for(month: str) -> str:
    return f"{BASE_URL}/lichess_db_standard_rated_{month}.pgn.zst"


def remote_size(url: str) -> int | None:
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            length = r.headers.get("Content-Length")
            return int(length) if length else None
    except urllib.error.URLError as e:
        print(f"HEAD fallita: {e}")
        return None


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def download(month: str, *, dest_dir: Path = DEST_DIR) -> int:
    url = url_for(month)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"lichess_db_standard_rated_{month}.pgn.zst"

    total = remote_size(url)
    if total is None:
        print("dimensione remota sconosciuta, procedo comunque")

    done = dest.stat().st_size if dest.exists() else 0
    if total is not None and done >= total:
        print(f"gia completo: {dest.name} ({human(done)})")
        return 0

    if done:
        print(f"riprendo da {human(done)}" + (f" / {human(total)}" if total else ""))
    else:
        print(f"scarico {dest.name}" + (f" ({human(total)})" if total else ""))

    headers = {"Range": f"bytes={done}-"} if done else {}
    req = urllib.request.Request(url, headers=headers)

    started = time.time()
    last_report = 0.0
    try:
        with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "ab") as fh:
            while True:
                chunk = resp.read(CHUNK)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)

                # Un report ogni 30 s: abbastanza per vedere che procede, non tanto da
                # riempire il log di un download che dura ore.
                now = time.time()
                if now - last_report >= 30:
                    elapsed = now - started
                    speed = done / elapsed if elapsed > 0 else 0
                    pct = f" {100 * done / total:.1f}%" if total else ""
                    eta = ""
                    if total and speed > 0:
                        eta = f", ETA {(total - done) / speed / 60:.0f} min"
                    print(f"  {human(done)}{pct} a {human(speed)}/s{eta}", flush=True)
                    last_report = now
    except (urllib.error.URLError, TimeoutError) as e:
        print(f"interrotto ({type(e).__name__}: {e}) — rilanciare per riprendere")
        return 1

    print(f"completato: {dest} ({human(dest.stat().st_size)})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--month", default="2026-07", help="mese del dump, formato YYYY-MM")
    ap.add_argument("--check", action="store_true", help="stampa la dimensione e esce")
    args = ap.parse_args()

    url = url_for(args.month)
    if args.check:
        size = remote_size(url)
        print(f"{url}\n  {human(size) if size else 'dimensione sconosciuta'}")
        return 0

    return download(args.month)


if __name__ == "__main__":
    sys.exit(main())
