"""Runner dei check della pipeline.

Implementa i tre livelli descritti in PIPELINE.md:

    python scripts/check.py --level 1       # pre-commit, < 20 s
    python scripts/check.py --level 2       # pre-merge, 2-5 min
    python scripts/check.py --stage setup   # gate di fase
    python scripts/check.py --all           # tutti i gate applicabili

Uscita 0 = verde, 1 = rosso. Nessun gate rosso va aggirato: si corregge, oppure si
documenta in docs/DECISIONS.md perche il criterio cambia.

I gate delle fasi non ancora implementate riportano SKIP, non PASS: un check che non
gira non e un check superato.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

GREEN, RED, YELLOW, GREY, BOLD, RESET = (
    "\033[32m",
    "\033[31m",
    "\033[33m",
    "\033[90m",
    "\033[1m",
    "\033[0m",
)


@dataclass
class Result:
    name: str
    status: str  # PASS | FAIL | SKIP
    detail: str = ""


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, **kwargs)


def has_python_sources() -> bool:
    """True se esiste almeno un modulo Python oltre agli script di supporto."""
    return any(SRC.rglob("*.py"))


def has_tests(subdir: str = "") -> bool:
    base = ROOT / "tests" / subdir if subdir else ROOT / "tests"
    return base.exists() and any(base.rglob("test_*.py"))


# --------------------------------------------------------------------------------------
# Livello 1 — pre-commit
# --------------------------------------------------------------------------------------


def check_ruff_lint() -> Result:
    if not has_python_sources():
        return Result("ruff lint", "SKIP", "nessun sorgente Python ancora")
    if shutil.which("ruff") is None:
        return Result("ruff lint", "FAIL", "ruff non installato (pip install -r requirements-dev.txt)")
    p = run(["ruff", "check", "src", "tests", "scripts"])
    return Result("ruff lint", "PASS" if p.returncode == 0 else "FAIL", p.stdout.strip()[:600])


def check_ruff_format() -> Result:
    if not has_python_sources():
        return Result("ruff format", "SKIP", "nessun sorgente Python ancora")
    if shutil.which("ruff") is None:
        return Result("ruff format", "FAIL", "ruff non installato")
    p = run(["ruff", "format", "--check", "src", "tests", "scripts"])
    return Result("ruff format", "PASS" if p.returncode == 0 else "FAIL", p.stdout.strip()[:600])


def check_mypy() -> Result:
    if not has_python_sources():
        return Result("mypy", "SKIP", "nessun sorgente Python ancora")
    if shutil.which("mypy") is None:
        return Result("mypy", "FAIL", "mypy non installato")
    p = run(["mypy", "src"])
    return Result("mypy", "PASS" if p.returncode == 0 else "FAIL", p.stdout.strip()[-800:])


def check_tests_fast() -> Result:
    if not has_tests():
        return Result("pytest -m fast", "SKIP", "nessun test ancora")
    p = run(["pytest", "-q", "-m", "fast", "--no-header"])
    # exit code 5 = nessun test raccolto con quel marker
    if p.returncode == 5:
        return Result("pytest -m fast", "SKIP", "nessun test marcato 'fast'")
    return Result("pytest -m fast", "PASS" if p.returncode == 0 else "FAIL", p.stdout.strip()[-800:])


def check_no_large_files() -> Result:
    """Impedisce che pesi, dataset e PGN finiscano in git (regola #5)."""
    p = run(["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"])
    staged = [f for f in p.stdout.split("\n") if f.strip()]
    if not staged:
        p = run(["git", "ls-files"])
        staged = [f for f in p.stdout.split("\n") if f.strip()]

    forbidden_ext = {".pt", ".pth", ".onnx", ".pgn", ".zst", ".npy", ".npz", ".bin", ".ckpt"}
    offenders: list[str] = []
    for f in staged:
        path = ROOT / f
        if path.suffix.lower() in forbidden_ext:
            offenders.append(f"{f} (estensione {path.suffix})")
        elif path.is_file() and path.stat().st_size > 5_000_000:
            offenders.append(f"{f} ({path.stat().st_size // 1_000_000} MB)")

    if offenders:
        return Result("nessun file grande in git", "FAIL", "; ".join(offenders[:10]))
    return Result("nessun file grande in git", "PASS")


# --------------------------------------------------------------------------------------
# Livello 2 — pre-merge
# --------------------------------------------------------------------------------------


def check_tests_all() -> Result:
    if not has_tests():
        return Result("pytest (suite completa)", "SKIP", "nessun test ancora")
    p = run(["pytest", "-q", "--no-header", "-m", "not slow"])
    if p.returncode == 5:
        return Result("pytest (suite completa)", "SKIP", "nessun test raccolto")
    return Result(
        "pytest (suite completa)", "PASS" if p.returncode == 0 else "FAIL", p.stdout.strip()[-1200:]
    )


def check_golden_files() -> Result:
    golden = ROOT / "tests" / "golden"
    files = list(golden.glob("*.json")) if golden.exists() else []
    if not files:
        return Result("file golden presenti", "SKIP", "da generare nello Stadio 1")
    return Result("file golden presenti", "PASS", f"{len(files)} file")


# --------------------------------------------------------------------------------------
# Gate 0 — setup
# --------------------------------------------------------------------------------------


def check_python_version() -> Result:
    v = sys.version_info
    ok = (v.major, v.minor) == (3, 11)
    return Result(
        "Python 3.11",
        "PASS" if ok else "FAIL",
        f"trovato {v.major}.{v.minor}.{v.micro}" + ("" if ok else " — attesa 3.11"),
    )


def check_venv_active() -> Result:
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    return Result(
        "venv attivo",
        "PASS" if in_venv else "FAIL",
        sys.prefix if in_venv else "usare .\\.venv\\Scripts\\Activate.ps1",
    )


def check_torch_cuda() -> Result:
    try:
        import torch
    except ImportError:
        return Result("torch + CUDA", "FAIL", "torch non installato — vedi docs/VENV.md")
    if not torch.cuda.is_available():
        return Result(
            "torch + CUDA",
            "FAIL",
            f"torch {torch.__version__} senza CUDA — probabile wheel CPU-only, reinstallare da --index-url cu126",
        )
    return Result("torch + CUDA", "PASS", f"torch {torch.__version__} su {torch.cuda.get_device_name(0)}")


def check_vram() -> Result:
    try:
        import torch
    except ImportError:
        return Result("VRAM >= 5 GB", "SKIP", "torch non installato")
    if not torch.cuda.is_available():
        return Result("VRAM >= 5 GB", "SKIP", "CUDA non disponibile")
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return Result(
        "VRAM >= 5 GB",
        "PASS" if total_gb >= 5.0 else "FAIL",
        f"{total_gb:.1f} GB",
    )


def check_core_imports() -> Result:
    missing: list[str] = []
    for mod in ("chess", "numpy", "yaml", "pydantic", "tqdm"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        return Result("import del nucleo", "FAIL", "mancanti: " + ", ".join(missing))
    return Result("import del nucleo", "PASS")


def check_config_loads() -> Result:
    cfg = ROOT / "configs" / "default.yaml"
    if not cfg.exists():
        return Result("configs/default.yaml", "FAIL", "file assente")
    try:
        import yaml

        data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    except ImportError:
        return Result("configs/default.yaml", "SKIP", "pyyaml non installato")
    except Exception as e:  # noqa: BLE001
        return Result("configs/default.yaml", "FAIL", f"YAML non valido: {e}")
    if not isinstance(data, dict) or "seed" not in data:
        return Result("configs/default.yaml", "FAIL", "manca la chiave 'seed'")
    return Result("configs/default.yaml", "PASS", f"{len(data)} sezioni")


def check_precommit_installed() -> Result:
    hook = ROOT / ".git" / "hooks" / "pre-commit"
    if not hook.exists():
        return Result("hook pre-commit installato", "FAIL", "eseguire: pre-commit install")
    return Result("hook pre-commit installato", "PASS")


def check_seed_reproducible() -> Result:
    """Due chiamate a set_seed con lo stesso valore devono dare la stessa sequenza."""
    try:
        sys.path.insert(0, str(SRC))
        from chessbot.utils.seed import set_seed
    except Exception:  # noqa: BLE001
        return Result("seed riproducibile", "SKIP", "chessbot.utils.seed non ancora implementato")

    import random

    set_seed(1234)
    a = [random.random() for _ in range(5)]
    set_seed(1234)
    b = [random.random() for _ in range(5)]
    return Result("seed riproducibile", "PASS" if a == b else "FAIL")


def check_stockfish() -> Result:
    """Non blocca il setup: serve dallo Stadio 2 in poi."""
    candidates = list((ROOT / "tools").rglob("stockfish*")) if (ROOT / "tools").exists() else []
    if shutil.which("stockfish") or candidates:
        return Result("Stockfish disponibile", "PASS")
    return Result(
        "Stockfish disponibile", "SKIP", "serve dallo Stadio 2 — scaricare in tools/stockfish/"
    )


# --------------------------------------------------------------------------------------
# Gate di fase non ancora implementati
# --------------------------------------------------------------------------------------


def _stage_placeholder(stage: str, module: str, tests_dir: str) -> list[Result]:
    """Gate delle fasi successive: SKIP finche il codice non esiste."""
    mod_path = SRC / "chessbot" / module
    # __init__.py non conta: un package vuoto non e un modulo implementato.
    implemented = mod_path.exists() and any(
        p.name != "__init__.py" for p in mod_path.glob("*.py")
    )
    if not implemented:
        return [Result(f"gate {stage}", "SKIP", f"src/chessbot/{module}/ non ancora implementato")]

    if not has_tests(tests_dir):
        return [
            Result(
                f"gate {stage}",
                "FAIL",
                f"codice presente in {module}/ ma nessun test in tests/{tests_dir}/ — regola #6",
            )
        ]
    p = run(["pytest", "-q", "--no-header", f"tests/{tests_dir}"])
    return [Result(f"gate {stage}", "PASS" if p.returncode == 0 else "FAIL", p.stdout.strip()[-1000:])]


# --------------------------------------------------------------------------------------
# Registro
# --------------------------------------------------------------------------------------

LEVEL_1 = [check_ruff_lint, check_ruff_format, check_mypy, check_tests_fast, check_no_large_files]
LEVEL_2 = LEVEL_1 + [check_tests_all, check_golden_files]

STAGES = {
    "setup": [
        check_python_version,
        check_venv_active,
        check_core_imports,
        check_torch_cuda,
        check_vram,
        check_config_loads,
        check_precommit_installed,
        check_seed_reproducible,
        check_stockfish,
    ],
    "encoding": lambda: _stage_placeholder("encoding", "encoding", "unit"),
    "baseline": lambda: _stage_placeholder("baseline", "baseline", "integration"),
    "data": lambda: _stage_placeholder("data", "data", "integration"),
    "train": lambda: _stage_placeholder("train", "training", "integration"),
    "mcts": lambda: _stage_placeholder("mcts", "search", "integration"),
    "rl-entry": lambda: _stage_placeholder("rl-entry", "training", "integration"),
}


def collect(checks) -> list[Result]:
    if callable(checks) and not isinstance(checks, list):
        return checks()
    return [c() for c in checks]


def report(title: str, results: list[Result]) -> int:
    print(f"\n{BOLD}{title}{RESET}")
    print("-" * len(title))
    failed = 0
    for r in results:
        if r.status == "PASS":
            mark, color = "PASS", GREEN
        elif r.status == "SKIP":
            mark, color = "SKIP", GREY
        else:
            mark, color = "FAIL", RED
            failed += 1
        line = f"  {color}[{mark}]{RESET} {r.name}"
        if r.detail:
            line += f"  {GREY}{r.detail.splitlines()[0][:100] if r.detail else ''}{RESET}"
        print(line)
        if r.status == "FAIL" and len(r.detail.splitlines()) > 1:
            for extra in r.detail.splitlines()[1:8]:
                print(f"         {GREY}{extra[:110]}{RESET}")

    skipped = sum(1 for r in results if r.status == "SKIP")
    passed = sum(1 for r in results if r.status == "PASS")
    print(f"\n  {passed} pass, {failed} fail, {skipped} skip")
    if failed:
        print(f"  {RED}Gate ROSSO — correggere prima di procedere (PIPELINE.md §0){RESET}")
    elif skipped and not passed:
        print(f"  {YELLOW}Nulla da verificare ancora.{RESET}")
    else:
        print(f"  {GREEN}Gate verde.{RESET}")
    return failed


def main() -> int:
    ap = argparse.ArgumentParser(description="Runner dei check della pipeline (vedi PIPELINE.md)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--level", type=int, choices=[1, 2], help="livello di check")
    g.add_argument("--stage", choices=sorted(STAGES), help="gate di fase")
    g.add_argument("--all", action="store_true", help="tutti i livelli e i gate")
    args = ap.parse_args()

    failed = 0
    if args.level == 1:
        failed += report("Livello 1 — pre-commit", collect(LEVEL_1))
    elif args.level == 2:
        failed += report("Livello 2 — pre-merge", collect(LEVEL_2))
    elif args.stage:
        failed += report(f"Gate — {args.stage}", collect(STAGES[args.stage]))
    else:
        failed += report("Livello 2 — pre-merge", collect(LEVEL_2))
        for name in STAGES:
            failed += report(f"Gate — {name}", collect(STAGES[name]))

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
