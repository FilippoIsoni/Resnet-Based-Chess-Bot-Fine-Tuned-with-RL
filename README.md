# Chess bot — ResNet + MCTS, fine-tuned con Expert Iteration

Motore scacchistico costruito hands-on applicando supervised learning e reinforcement
learning. Obiettivo di livello medio-alto: **la forza massima non e il fine**, la
comprensione del percorso si.

**Stack:** Python 3.11 + PyTorch (CUDA 12.6) · **Hardware:** RTX 3050 Laptop, 6 GB VRAM.

## Stato

**Il motore è finito e misurato: Elo 1964 ± 62** contro Stockfish a forza limitata
(`UCI_LimitStrength`). ResNet 8×128 da 12,0 M parametri, addestrata su 20 M posizioni
Lichess e poi affinata con Expert Iteration (25 iterazioni, +119 ± 51 Elo). Tutti i gate
da 0 a 6 sono verdi.

In corso lo **Stadio 7**: renderlo giocabile da browser. Il lavoro è diviso in due metà
indipendenti — vedi sotto.

## Documenti

| File | Contenuto |
|---|---|
| [piano-motore-scacchi.md](piano-motore-scacchi.md) | **Il cosa** — architettura, fasi, criticita tecniche |
| [PIPELINE.md](PIPELINE.md) | **Il come** — ordine di lavoro, gate di verifica, regole permanenti |
| [docs/API_CONTRACT.md](docs/API_CONTRACT.md) | **Il contratto** fra UI e backend — da leggere prima di toccare lo Stadio 7 |
| [docs/VENV.md](docs/VENV.md) | Ambiente: cosa contiene e perche |
| [docs/DECISIONS.md](docs/DECISIONS.md) | Scelte non ovvie e deviazioni dal piano |
| [docs/RESULTS.md](docs/RESULTS.md) | Registro delle misure, fallimenti compresi |
| [docs/JOURNAL.md](docs/JOURNAL.md) | Diario dei bug che sono costati tempo |

## Setup

### Windows + NVIDIA (macchina di riferimento)

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

### macOS — da verificare (Stadio 0-bis)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

pip install torch                        # niente --index-url: su Mac non ci sono wheel CUDA
pip install -r requirements-dev.txt
pip install -e .

pre-commit install
python scripts/check.py --stage setup
```

I check gestiscono già l'assenza di CUDA (i criteri specifici diventano SKIP, non FAIL),
ma **questo va confermato girandoli davvero su Mac**: è il prossimo passo assegnato al
collaboratore su quella macchina. Dettagli in [docs/VENV.md](docs/VENV.md#macos--da-verificare-stadio-0-bis)
e [PIPELINE.md](PIPELINE.md).

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

In parallelo, lo **Stadio 0-bis**: verifica di compatibilità macOS, assegnata al
collaboratore su Mac. Va chiuso prima dello Stadio 1, così i file golden nascono già
validati su entrambe le piattaforme.

## Stadio 7 — giocarci dal browser

Il motore esiste ma è raggiungibile solo da terminale. Lo Stadio 7 lo mette online, e il
vincolo che decide tutto è che **GitHub Pages serve solo file statici**: non esegue Python.
Quindi l'interfaccia compilata sta su Pages e il motore gira altrove come servizio HTTP.

```
   Browser
      │
      ├── GitHub Pages ──────► UI Flutter (statica)          [Filippo]
      │                             │
      │                             │  POST /move {fen, level}
      │                             ▼
      └── Hugging Face Spaces ─► FastAPI + ResNet + MCTS     [collaboratore Mac]
```

| Metà | Chi | Cosa | Dove |
|---|---|---|---|
| Frontend | Filippo | UI Flutter Web, deploy su Pages | `web/` |
| Backend | collaboratore su Mac | server FastAPI, deploy su HF Spaces | `src/chessbot/api/`, `deploy/` |

Le due metà si sviluppano **in parallelo e senza dipendere l'una dall'altra**: la UI si
costruisce contro un motore finto che restituisce mosse casuali, il backend si verifica con
`curl`. L'unico punto di accordo è [docs/API_CONTRACT.md](docs/API_CONTRACT.md), che va
modificato *prima* di cambiare il proprio lato.

Tre livelli di difficoltà, che sono lo stesso motore con profondità di ricerca diversa:
50 simulazioni (~1500 Elo stimato), 200 (**1964 ± 62 misurato**), 800 (~2100 stimato).

### Sviluppo della UI

```bash
cd web
flutter pub get
flutter analyze && flutter test
flutter run -d chrome                    # motore finto, non serve il backend
```

## Il principio

In questo progetto i bug non crashano. Un encoder che dimentica di specchiare l'indice
della mossa, un segno invertito nel backup dell'MCTS, uno split del dataset fatto per
posizione invece che per partita: nessuno di questi produce un'eccezione. Producono un
motore mediocre di cui ti accorgi due settimane dopo, guardando una metrica che non sale.

Per questo la pipeline e costruita attorno alla verifica, e **nessuno stadio avanza
senza il suo gate**.
