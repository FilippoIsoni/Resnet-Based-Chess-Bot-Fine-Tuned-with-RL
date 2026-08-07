# Chess bot — ResNet + MCTS, fine-tuned con Expert Iteration

Motore scacchistico costruito hands-on applicando supervised learning e reinforcement
learning. Obiettivo di livello medio-alto: **la forza massima non e il fine**, la
comprensione del percorso si.

**Stack:** Python 3.11 + PyTorch (CUDA 12.6) · **Hardware:** RTX 3050 Laptop, 6 GB VRAM.

## Documenti

| File | Contenuto |
|---|---|
| [piano-motore-scacchi.md](piano-motore-scacchi.md) | **Il cosa** — architettura, fasi, criticita tecniche |
| [PIPELINE.md](PIPELINE.md) | **Il come** — ordine di lavoro, gate di verifica, regole permanenti |
| [docs/VENV.md](docs/VENV.md) | Ambiente: cosa contiene e perche |
| [docs/DECISIONS.md](docs/DECISIONS.md) | Scelte non ovvie e deviazioni dal piano |
| [docs/RESULTS.md](docs/RESULTS.md) | Registro delle misure, fallimenti compresi |
| [docs/JOURNAL.md](docs/JOURNAL.md) | Diario dei bug che sono costati tempo |

## Setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# torch PRIMA e dall'indice CUDA, altrimenti pip risolve la variante CPU-only
pip install torch==2.13.0+cu126 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements-dev.txt
pip install -e .

pre-commit install
python scripts/check.py --stage setup    # deve essere verde prima di iniziare
```

## Uso quotidiano

```powershell
python scripts/check.py --level 1        # pre-commit, < 20 s (automatico via hook)
python scripts/check.py --level 2        # prima di chiudere un pezzo di lavoro
python scripts/check.py --stage encoding # gate di fase
python scripts/preflight.py              # prima di ogni sessione lunga di training
```

## Le cinque fasi

| # | Fase | Elo atteso | Gate |
|---|---|---|---|
| 1 | Baseline minimax | ~1300 | `--stage baseline` |
| 2 | Preprocessing | — | `--stage data` |
| 3 | Training supervisionato | 1700-1900 | `--stage train` |
| 4 | MCTS | 2000-2300 | `--stage mcts` → **bot funzionante** |
| 5 | Reinforcement Learning | +100-250 | `--stage rl-entry` |

Prima di tutte: lo **Stadio 1**, encoder e codifica mosse (`--stage encoding`). Tutto il
resto ci si appoggia.

## Il principio

In questo progetto i bug non crashano. Un encoder che dimentica di specchiare l'indice
della mossa, un segno invertito nel backup dell'MCTS, uno split del dataset fatto per
posizione invece che per partita: nessuno di questi produce un'eccezione. Producono un
motore mediocre di cui ti accorgi due settimane dopo, guardando una metrica che non sale.

Per questo la pipeline e costruita attorno alla verifica, e **nessuno stadio avanza
senza il suo gate**.
