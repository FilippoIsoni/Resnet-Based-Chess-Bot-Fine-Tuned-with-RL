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
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

# La console Windows di default non e UTF-8: senza questo gli accenti escono come "?".
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
GREEN, RED, YELLOW, GREY, BOLD, RESET = (
    ("\033[32m", "\033[31m", "\033[33m", "\033[90m", "\033[1m", "\033[0m")
    if _COLOR
    else ("", "", "", "", "", "")
)


@dataclass
class Result:
    name: str
    status: str  # PASS | FAIL | SKIP
    detail: str = ""


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", **kwargs
    )


@lru_cache(maxsize=1)
def project_python() -> str:
    """Interprete che ha i tool del progetto installati.

    Tre contesti, tre risposte diverse:
    - shell col venv attivo o CI      -> `sys.executable` va bene
    - hook pre-commit                 -> gira in un venv ISOLATO che non ha ruff/mypy/pytest
    - `python scripts/check.py` a mano -> interprete di sistema, senza i tool

    Negli ultimi due casi bisogna risolvere esplicitamente il venv del progetto, altrimenti
    i check falliscono con "tool non installato" mentre i tool ci sono eccome.
    """
    if _has_module(sys.executable, "ruff"):
        return sys.executable

    for candidate in (ROOT / ".venv" / "Scripts" / "python.exe", ROOT / ".venv" / "bin" / "python"):
        if candidate.exists():
            return str(candidate)

    return sys.executable


def _has_module(python: str, module: str) -> bool:
    p = subprocess.run(
        [
            python,
            "-c",
            f"import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('{module}') else 1)",
        ],
        capture_output=True,
    )
    return p.returncode == 0


def run_module(module: str, *args: str) -> subprocess.CompletedProcess:
    """Invoca un tool come modulo dell'interprete del progetto.

    Non `subprocess.run(["ruff", ...])`: quello cerca l'eseguibile nel PATH, che fuori da
    un venv attivo non contiene i tool del progetto.
    """
    return run([project_python(), "-m", module, *args])


def module_available(module: str) -> bool:
    return _has_module(project_python(), module)


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
    if not module_available("ruff"):
        return Result(
            "ruff lint", "FAIL", "ruff non installato (pip install -r requirements-dev.txt)"
        )
    p = run_module("ruff", "check", "src", "tests", "scripts")
    return Result("ruff lint", "PASS" if p.returncode == 0 else "FAIL", p.stdout.strip()[:600])


def check_ruff_format() -> Result:
    if not has_python_sources():
        return Result("ruff format", "SKIP", "nessun sorgente Python ancora")
    if not module_available("ruff"):
        return Result("ruff format", "FAIL", "ruff non installato")
    p = run_module("ruff", "format", "--check", "src", "tests", "scripts")
    return Result("ruff format", "PASS" if p.returncode == 0 else "FAIL", p.stdout.strip()[:600])


def check_mypy() -> Result:
    if not has_python_sources():
        return Result("mypy", "SKIP", "nessun sorgente Python ancora")
    if not module_available("mypy"):
        return Result("mypy", "FAIL", "mypy non installato")
    p = run_module("mypy", "src")
    return Result("mypy", "PASS" if p.returncode == 0 else "FAIL", p.stdout.strip()[-800:])


def check_tests_fast() -> Result:
    if not has_tests():
        return Result("pytest -m fast", "SKIP", "nessun test ancora")
    p = run_module("pytest", "-q", "-m", "fast", "--no-header")
    # exit code 5 = nessun test raccolto con quel marker
    if p.returncode == 5:
        return Result("pytest -m fast", "SKIP", "nessun test marcato 'fast'")
    return Result(
        "pytest -m fast", "PASS" if p.returncode == 0 else "FAIL", p.stdout.strip()[-800:]
    )


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
    p = run_module("pytest", "-q", "--no-header", "-m", "not slow")
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
    activate = (
        "source .venv/bin/activate"
        if sys.platform != "win32"
        else ".\\.venv\\Scripts\\Activate.ps1"
    )
    return Result(
        "venv attivo",
        "PASS" if in_venv else "FAIL",
        sys.prefix if in_venv else f"usare {activate}",
    )


def check_torch_accelerator() -> Result:
    """Verifica che torch abbia un acceleratore utilizzabile.

    Cross-platform: su Windows/Linux ci si aspetta CUDA, su macOS MPS (Apple Silicon).
    Un criterio che non si applica alla piattaforma non e un criterio fallito — vedi
    Stadio 0-bis in PIPELINE.md.
    """
    try:
        import torch
    except ImportError:
        return Result("torch + acceleratore", "FAIL", "torch non installato — vedi docs/VENV.md")

    if torch.cuda.is_available():
        return Result(
            "torch + acceleratore",
            "PASS",
            f"torch {torch.__version__}, CUDA su {torch.cuda.get_device_name(0)}",
        )

    if sys.platform == "darwin":
        if torch.backends.mps.is_available():
            return Result("torch + acceleratore", "PASS", f"torch {torch.__version__}, MPS")
        return Result(
            "torch + acceleratore",
            "SKIP",
            f"torch {torch.__version__} su CPU — Mac Intel: atteso, training pesante sulla macchina CUDA",
        )

    return Result(
        "torch + acceleratore",
        "FAIL",
        f"torch {torch.__version__} senza CUDA — probabile wheel CPU-only, reinstallare da --index-url cu126",
    )


def check_vram() -> Result:
    try:
        import torch
    except ImportError:
        return Result("memoria acceleratore >= 5 GB", "SKIP", "torch non installato")

    if torch.cuda.is_available():
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return Result(
            "memoria acceleratore >= 5 GB",
            "PASS" if total_gb >= 5.0 else "FAIL",
            f"{total_gb:.1f} GB VRAM",
        )

    if sys.platform == "darwin":
        # Su Apple Silicon la memoria e unificata: non c'e una VRAM dedicata da misurare,
        # il limite e la RAM di sistema. Il criterio non si applica.
        return Result(
            "memoria acceleratore >= 5 GB",
            "SKIP",
            "memoria unificata su macOS, criterio non applicabile",
        )

    return Result("memoria acceleratore >= 5 GB", "SKIP", "nessun acceleratore CUDA")


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
    except Exception as e:
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
    except Exception:
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


def check_encoding_gate() -> list[Result]:
    """Gate 1 — encoder e codifica mosse (§2.5).

    Non delega alla suite intera: elenca i criteri del piano uno per uno, cosi il
    report dice QUALE criterio e rosso invece di un generico "pytest fallito".
    """
    mod = SRC / "chessbot" / "encoding"
    if not any(p.name != "__init__.py" for p in mod.glob("*.py")):
        return [Result("gate encoding", "SKIP", "src/chessbot/encoding/ non ancora implementato")]

    results: list[Result] = []

    # Il golden deve coincidere con cio che l'encoder produce adesso. E anche il
    # confronto Windows/macOS dello Stadio 0-bis.
    p = run([project_python(), "scripts/gen_golden.py", "--check"])
    results.append(
        Result(
            "golden invariato",
            "PASS" if p.returncode == 0 else "FAIL",
            (p.stdout + p.stderr).strip()[:400],
        )
    )

    # I criteri di §2.5, mappati sui test che li verificano.
    criteri = {
        "20 posizioni golden": "test_golden_corrisponde_all_encoder_attuale",
        "round-trip mossa<->indice": "test_round_trip_su_mosse_legali",
        "simmetria specchiata (criticita #1)": "test_simmetria_specchiata_tensore_e_indice",
        "promozione a donna (criticita #4)": "test_promozione_a_donna_nel_blocco_mosse_da_regina",
        "somma piani == n. pezzi": "test_somma_piani_pezzi_uguale_numero_pezzi",
        "encoder vs re-parsing FEN": "test_ricostruzione_fen_dal_tensore",
        "nessun indice fuori [0, 4671]": "test_nessun_indice_fuori_range_su_tutto_il_campione",
    }
    for etichetta, test_name in criteri.items():
        p = run_module("pytest", "-q", "--no-header", "tests/unit", "-k", test_name)
        if p.returncode == 5:
            results.append(Result(etichetta, "FAIL", f"test '{test_name}' non trovato"))
        else:
            results.append(
                Result(
                    etichetta,
                    "PASS" if p.returncode == 0 else "FAIL",
                    "" if p.returncode == 0 else p.stdout.strip()[-500:],
                )
            )

    # La suite completa dell'encoding, per cio che i criteri sopra non nominano.
    p = run_module("pytest", "-q", "--no-header", "tests/unit")
    results.append(
        Result(
            "suite tests/unit completa",
            "PASS" if p.returncode == 0 else "FAIL",
            p.stdout.strip()[-600:],
        )
    )
    return results


def check_baseline_gate() -> list[Result]:
    """Gate 2 — baseline minimax (§3, Stadio 2).

    Il criterio "batte random 100/100" e lento: qui si verifica il risultato salvato da
    `scripts/run_gate2_match.py`, invece di rigiocare il match ad ogni invocazione. Un
    risultato senza file e un criterio non verificato, quindi SKIP con l'istruzione per
    produrlo — non PASS.
    """
    mod = SRC / "chessbot" / "baseline"
    if not any(p.name != "__init__.py" for p in mod.glob("*.py")):
        return [Result("gate baseline", "SKIP", "src/chessbot/baseline/ non ancora implementato")]

    results: list[Result] = []

    criteri = {
        "perft su posizioni standard": ("tests/unit", "test_perft_fino_a_profondita_3"),
        "valutazione simmetrica (criticita #2)": (
            "tests/unit",
            "test_invarianza_per_specchiatura_completa or test_antisimmetria_per_cambio_di_tratto",
        ),
        "matto in 1 su 50 posizioni": ("tests/unit", "test_trova_matto_in_1_su_50_posizioni"),
    }
    for etichetta, (path, expr) in criteri.items():
        p = run_module("pytest", "-q", "--no-header", path, "-k", expr)
        if p.returncode == 5:
            results.append(Result(etichetta, "FAIL", f"nessun test corrisponde a '{expr}'"))
        else:
            results.append(
                Result(
                    etichetta,
                    "PASS" if p.returncode == 0 else "FAIL",
                    "" if p.returncode == 0 else p.stdout.strip()[-500:],
                )
            )

    # Perft profondo e match brevi: marcati slow, esclusi dai livelli 1 e 2.
    p = run_module("pytest", "-q", "--no-header", "tests/integration", "-m", "slow")
    if p.returncode == 5:
        results.append(Result("perft profondo e match brevi", "SKIP", "nessun test slow"))
    else:
        results.append(
            Result(
                "perft profondo e match brevi",
                "PASS" if p.returncode == 0 else "FAIL",
                p.stdout.strip()[-600:],
            )
        )

    results.append(_check_gate2_match_result())
    return results


def _check_gate2_match_result() -> Result:
    """Legge l'ultimo match salvato in runs/gate2/ e ne verifica il criterio."""
    import json

    runs = ROOT / "runs" / "gate2"
    files = sorted(runs.glob("match_random_*.json")) if runs.exists() else []
    if not files:
        return Result(
            "batte random 100/100",
            "SKIP",
            "nessun match registrato — eseguire: python scripts/run_gate2_match.py",
        )

    try:
        data = json.loads(files[-1].read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return Result("batte random 100/100", "FAIL", f"{files[-1].name} illeggibile: {e}")

    wins, games = data.get("wins", 0), data.get("games", 0)
    detail = f"{data.get('summary', '')} [commit {data.get('commit', '?')}]"
    if games < 100:
        return Result("batte random 100/100", "FAIL", f"solo {games} partite giocate. {detail}")
    return Result("batte random 100/100", "PASS" if wins == games else "FAIL", detail)


def check_data_gate() -> list[Result]:
    """Gate 3 — pipeline dati (§3, Stadio 3).

    Due meta distinte:
    - la LOGICA si verifica con i test su PGN sintetici, che girano sempre
    - il DATASET prodotto si verifica con `scripts/verify_dataset.py`, che legge gli
      shard veri. Senza dataset e SKIP con l'istruzione per produrlo, non PASS: un
      criterio che non gira non e un criterio superato.
    """
    mod = SRC / "chessbot" / "data"
    if not any(p.name != "__init__.py" for p in mod.glob("*.py")):
        return [Result("gate data", "SKIP", "src/chessbot/data/ non ancora implementato")]

    results: list[Result] = []

    criteri = {
        "split per PARTITA, 0 game-id condivisi (criticita #3)": (
            "test_zero_game_id_condivisi_fra_split or test_le_partite_restano_intere"
        ),
        "0 FEN condivise fra split": "test_zero_fen_condivise_fra_split",
        "il controllo anti-leakage sa fallire": "test_lo_split_per_posizione_verrebbe_intercettato",
        "capping aperture": "test_capping_aperture",
        "round-trip storage bit a bit": (
            "test_round_trip_pack_unpack_bit_a_bit or test_shard_scritto_e_riletto_identico"
        ),
        "filtri Elo / time control / esito": (
            "test_accept_headers_scarta or test_categoria_dal_time_control"
        ),
        "mosse legali nelle posizioni estratte": "test_ogni_mossa_e_legale_nella_sua_posizione",
    }
    for etichetta, expr in criteri.items():
        p = run_module("pytest", "-q", "--no-header", "tests/unit", "-k", expr)
        if p.returncode == 5:
            results.append(Result(etichetta, "FAIL", f"nessun test corrisponde a '{expr}'"))
        else:
            results.append(
                Result(
                    etichetta,
                    "PASS" if p.returncode == 0 else "FAIL",
                    "" if p.returncode == 0 else p.stdout.strip()[-500:],
                )
            )

    results.append(_check_built_dataset())
    results.append(_check_alignment())
    return results


def _check_alignment() -> Result:
    """Allineamento posizione <-> mossa sul dataset costruito.

    Su un campione: la scansione completa dei 20M dura 18 minuti, troppo per un gate che
    si lancia spesso. Il risultato completo e in docs/RESULTS.md e si riproduce con
    `python scripts/verify_alignment.py --data data/processed`.
    """
    processed = ROOT / "data" / "processed"
    if not (processed / "manifest.json").exists():
        return Result("allineamento posizione-mossa", "SKIP", "nessun dataset costruito")

    p = run(
        [
            project_python(),
            "scripts/verify_alignment.py",
            "--data",
            str(processed),
            "--limit",
            "50000",
        ]
    )
    detail = ""
    for line in (p.stdout or "").splitlines():
        if line.strip().startswith("ok "):
            detail = " ".join(line.split())
            break
    return Result(
        "allineamento posizione-mossa",
        "PASS" if p.returncode == 0 else "FAIL",
        detail + " (campione 50k per split)" if detail else (p.stdout or "")[-300:],
    )


def _check_built_dataset() -> Result:
    """Verifica il dataset costruito, se esiste."""
    processed = ROOT / "data" / "processed"
    manifest = processed / "manifest.json"
    if not manifest.exists():
        return Result(
            "dataset costruito e verificato",
            "SKIP",
            "nessun dataset — eseguire: python scripts/build_dataset.py --input <dump>",
        )

    # Solo i primi shard per split: rileggere i 20M interi costa ~25 minuti, troppo per
    # un gate che si lancia spesso. La verifica completa e comunque stata fatta una volta
    # e registrata in docs/RESULTS.md; si rilancia esplicitamente, senza il limite, prima
    # di un training vero.
    p = run(
        [
            project_python(),
            "scripts/verify_dataset.py",
            "--data",
            str(processed),
            "--shards-per-split",
            "2",
        ]
    )
    detail = ""
    for line in (p.stdout or "").splitlines():
        if "totale:" in line:
            detail = line.strip()
    return Result(
        "dataset costruito e verificato",
        "PASS" if p.returncode == 0 else "FAIL",
        detail or (p.stdout or p.stderr).strip()[-400:],
    )


def _stage_placeholder(stage: str, module: str, tests_dir: str) -> list[Result]:
    """Gate delle fasi successive: SKIP finche il codice non esiste."""
    mod_path = SRC / "chessbot" / module
    # __init__.py non conta: un package vuoto non e un modulo implementato.
    implemented = mod_path.exists() and any(p.name != "__init__.py" for p in mod_path.glob("*.py"))
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
    p = run_module("pytest", "-q", "--no-header", f"tests/{tests_dir}")
    return [
        Result(f"gate {stage}", "PASS" if p.returncode == 0 else "FAIL", p.stdout.strip()[-1000:])
    ]


# --------------------------------------------------------------------------------------
# Registro
# --------------------------------------------------------------------------------------

LEVEL_1 = [check_ruff_lint, check_ruff_format, check_mypy, check_tests_fast, check_no_large_files]
LEVEL_2 = [*LEVEL_1, check_tests_all, check_golden_files]

STAGES = {
    "setup": [
        check_python_version,
        check_venv_active,
        check_core_imports,
        check_torch_accelerator,
        check_vram,
        check_config_loads,
        check_precommit_installed,
        check_seed_reproducible,
        check_stockfish,
    ],
    "encoding": check_encoding_gate,
    "baseline": check_baseline_gate,
    "data": check_data_gate,
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
