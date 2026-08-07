"""Controlli da eseguire PRIMA di ogni sessione lunga di training (regola #7).

    python scripts/preflight.py

Non verifica la correttezza del codice — quello e compito di check.py. Verifica che la
macchina sia in condizione di reggere ore di lavoro senza buttare via la sessione:
spazio disco, VRAM libera, temperatura, checkpoint dir scrivibile.

Il piano lo dice esplicitamente: su un portatile un aggiornamento di sistema o uno
shutdown termico azzerano una notte di training.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
GREEN, RED, YELLOW, GREY, BOLD, RESET = (
    ("\033[32m", "\033[31m", "\033[33m", "\033[90m", "\033[1m", "\033[0m")
    if _COLOR
    else ("", "", "", "", "", "")
)

MIN_DISK_GB = 20.0
MIN_FREE_VRAM_GB = 4.0
MAX_GPU_TEMP_C = 75


def line(status: str, name: str, detail: str = "") -> bool:
    color = {"OK": GREEN, "WARN": YELLOW, "FAIL": RED}[status]
    print(f"  {color}[{status:4}]{RESET} {name}" + (f"  {GREY}{detail}{RESET}" if detail else ""))
    return status != "FAIL"


def check_disk() -> bool:
    free_gb = shutil.disk_usage(ROOT).free / 1024**3
    if free_gb < MIN_DISK_GB:
        return line("FAIL", "spazio disco", f"{free_gb:.1f} GB liberi, minimo {MIN_DISK_GB:.0f}")
    if free_gb < MIN_DISK_GB * 2:
        return line("WARN", "spazio disco", f"{free_gb:.1f} GB liberi")
    return line("OK", "spazio disco", f"{free_gb:.1f} GB liberi")


def nvidia_query(fields: str) -> list[str] | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return [v.strip() for v in out.stdout.strip().split("\n")[0].split(",")]


def check_gpu() -> bool:
    vals = nvidia_query("memory.total,memory.used,temperature.gpu,power.limit")
    if vals is None:
        if sys.platform == "darwin":
            # Su Mac non esiste nvidia-smi e la memoria e unificata: non c'e VRAM dedicata
            # da controllare. Il criterio non si applica, quindi non e un fallimento.
            return line(
                "WARN",
                "GPU",
                "nvidia-smi assente (macOS) — memoria unificata, controllo non applicabile",
            )
        return line("FAIL", "GPU", "nvidia-smi non disponibile")

    total, used, temp, plimit = float(vals[0]), float(vals[1]), float(vals[2]), vals[3]
    free_gb = (total - used) / 1024

    ok = True
    if free_gb < MIN_FREE_VRAM_GB:
        ok &= line(
            "FAIL",
            "VRAM libera",
            f"{free_gb:.1f} GB (minimo {MIN_FREE_VRAM_GB:.0f}) — chiudere browser e altre app GPU",
        )
    else:
        line("OK", "VRAM libera", f"{free_gb:.1f} GB su {total / 1024:.1f}")

    if temp > MAX_GPU_TEMP_C:
        line("WARN", "temperatura GPU", f"{temp:.0f} C a riposo — sollevare il portatile")
    else:
        line("OK", "temperatura GPU", f"{temp:.0f} C")

    line("OK", "power limit", f"{plimit} W  {GREY}(nvidia-smi -pl per abbassarlo){RESET}")
    return ok


def check_torch() -> bool:
    try:
        import torch
    except ImportError:
        return line("FAIL", "torch", "non installato")
    if torch.cuda.is_available():
        return line("OK", "torch CUDA", f"{torch.__version__} su {torch.cuda.get_device_name(0)}")
    if sys.platform == "darwin":
        if torch.backends.mps.is_available():
            return line("OK", "torch MPS", f"{torch.__version__} su Apple Silicon")
        return line(
            "WARN", "torch CPU", f"{torch.__version__} — training pesante sulla macchina CUDA"
        )
    return line("FAIL", "torch CUDA", "wheel CPU-only — vedi docs/VENV.md")


def check_checkpoint_dir() -> bool:
    runs = ROOT / "runs"
    try:
        runs.mkdir(exist_ok=True)
        probe = runs / ".preflight_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as e:
        return line("FAIL", "runs/ scrivibile", str(e))
    return line("OK", "runs/ scrivibile")


def check_git_clean() -> bool:
    """Un run non riproducibile non e un risultato (regola #2)."""
    p = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True)
    dirty = [x for x in p.stdout.split("\n") if x.strip()]
    if dirty:
        return line(
            "WARN",
            "working tree pulito",
            f"{len(dirty)} file modificati — il commit hash nel checkpoint non bastera a riprodurre",
        )
    rev = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()
    return line("OK", "working tree pulito", f"HEAD {rev}")


def check_power_plugged() -> bool:
    """Una sessione notturna a batteria si interrompe a meta. Vale su entrambe le piattaforme."""
    if sys.platform == "darwin":
        cmd = ["pmset", "-g", "batt"]
    elif sys.platform == "win32":
        # BatteryStatus: 1 = a batteria, 2 = alimentazione esterna
        cmd = [
            "powershell",
            "-NoProfile",
            "-Command",
            "(Get-CimInstance Win32_Battery).BatteryStatus",
        ]
    else:
        return line("WARN", "alimentazione", "controllo non implementato su questa piattaforma")

    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return line("WARN", "alimentazione", "stato non determinabile")

    on_battery = "Battery Power" in out if sys.platform == "darwin" else out.startswith("1")
    if on_battery:
        return line("WARN", "alimentazione", "a batteria — collegare il caricatore")
    return line("OK", "alimentazione", "collegata")


def main() -> int:
    print(f"\n{BOLD}Preflight — sessione di training{RESET}")
    print("-" * 33)
    results = [
        check_torch(),
        check_gpu(),
        check_disk(),
        check_checkpoint_dir(),
        check_power_plugged(),
        check_git_clean(),
    ]
    if all(results):
        print(f"\n  {GREEN}Pronto per il training.{RESET}\n")
        return 0
    print(f"\n  {RED}Preflight fallito — risolvere prima di lanciare.{RESET}\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
