# Ambiente virtuale — contenuto e motivazioni

Target: **Python 3.11**, Windows 11, RTX 3050 Laptop 6 GB, driver 566.14 (supporta CUDA 12.x).

> Python 3.11 e non 3.9 (l'altra installazione presente sul sistema): le wheel PyTorch CUDA
> recenti hanno smesso di essere pubblicate per 3.9, e 3.11 dà un interprete più veloce —
> non irrilevante, visto che il collo di bottiglia dell'MCTS sarà python-chess in puro Python
> (criticità #6).

## Creazione

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

PyTorch va installato **per primo e dal suo indice CUDA**, altrimenti pip risolve la variante
CPU-only e te ne accorgi solo quando il training va 40 volte più lento:

```powershell
pip install torch==2.13.0+cu126 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements-dev.txt
pip install -e .
```

Perche **cu126** e non cu124: l'indice cu124 e fermo a torch 2.6, mentre cu126 arriva a 2.13.
Il driver 566.14 installato supporta CUDA 12.6, quindi non c'e motivo di restare indietro.
La 3050 e architettura Ampere (sm_86), coperta da tutte le build recenti.

`requirements-dev.txt` include gia `requirements.txt`, quindi in sviluppo basta quello.

Verifica immediata, non rimandabile:

```powershell
python scripts/check.py --stage setup
```

---

## Cosa deve contenere e perché

### Nucleo — indispensabile da subito

| Pacchetto | Ruolo | Fase |
|---|---|---|
| `torch` (build cu124) | rete, training, inferenza | 3-5 |
| `python-chess` | regole, PGN, FEN, generazione mosse, Polyglot | tutte |
| `numpy` | tensori compatti, storage `.npy` | 2-5 |
| `pyyaml` | file di configurazione unico (§Fase 1) | tutte |
| `pydantic` | **validazione dello schema di config** | tutte |
| `tqdm` | barre di avanzamento su job da ore | 2-5 |

Su `pydantic`: un typo in un YAML non annotato non dà errore, dà un iperparametro di default
diverso da quello che credi di aver impostato. Con uno schema tipizzato il run si rifiuta di
partire. Costa dieci minuti e chiude un'intera classe di bug silenziosi.

### Dati e preprocessing — Fase 2

| Pacchetto | Ruolo |
|---|---|
| `zstandard` | i dump Lichess sono `.pgn.zst`, si leggono in streaming senza decomprimere 200 GB |
| `numpy` | shard memory-mapped (`np.memmap`) per la variante ottimizzata |
| `pyarrow` | *opzionale*, solo se si passa a Parquet per i metadati delle partite |

`multiprocessing` è in stdlib: nessuna dipendenza per il Pool a 6-8 worker.

### Training e monitoraggio — Fase 3

| Pacchetto | Ruolo |
|---|---|
| `tensorboard` | curve di loss, locale, zero configurazione |
| `wandb` | *opzionale*, alternativa cloud con confronto fra run |

Uno dei due è obbligatorio: il piano lo richiede esplicitamente. `tensorboard` è il default
perché non richiede account né rete.

`torch.amp` per il mixed precision è dentro torch, non serve `apex`.

### Valutazione — dalla Fase 3 in poi

| Pacchetto / binario | Ruolo |
|---|---|
| **Stockfish** (binario, non pip) | metro di riferimento con `UCI_LimitStrength` |
| `python-chess` | già include il client UCI (`chess.engine`) per parlarci |
| `scipy` | intervalli di confidenza, SPRT del gating (Fase 5) |

Stockfish si scarica da stockfishchess.org e si mette in `tools/stockfish/`, con il percorso
in `configs/default.yaml`. Non è un pacchetto Python.

`scipy` serve davvero solo alla Fase 5, ma è piccolo e lo si installa subito per non
frammentare l'ambiente.

### Deployment — Appendice B

| Pacchetto | Ruolo |
|---|---|
| `fastapi` | endpoint `POST /move`, `GET /health` |
| `uvicorn` | server ASGI |
| `onnx`, `onnxruntime` | export ed esecuzione CPU senza spedire 200 MB di PyTorch |

Questi vanno in un **requirements separato** (`requirements-serve.txt`): l'immagine di
deployment non deve contenere torch, altrimenti si vanifica il motivo per cui si esporta in ONNX.

### Sviluppo e check — la parte che tiene in piedi la pipeline

| Pacchetto | Ruolo |
|---|---|
| `pytest` | tutta la batteria di test |
| `pytest-cov` | copertura, per sapere cosa *non* è testato |
| `pytest-xdist` | test in parallelo: la batteria dell'encoder gira su 10.000 posizioni |
| `hypothesis` | **property-based testing** |
| `ruff` | lint + format, sostituisce flake8/black/isort |
| `mypy` | type checking statico |
| `pre-commit` | esegue L1 automaticamente ad ogni commit |

Su `hypothesis`: il round-trip `move → indice → move` è esattamente il caso d'uso per cui il
property testing esiste. Invece di scegliere a mano 10.000 mosse, si dichiara la proprietà
("per ogni mossa legale, decodificare l'indice restituisce la mossa originale") e la libreria
cerca attivamente il controesempio — e quando lo trova lo riduce al caso minimo. La promozione
a donna confusa con la sottopromozione (criticità #4) è il tipo di bug che hypothesis trova
in pochi secondi e una lista scritta a mano può mancare.

### Profiling — quando serve, non prima

| Pacchetto | Ruolo |
|---|---|
| `py-spy` | profiler a campionamento, si attacca a un processo già in esecuzione |
| `snakeviz` | visualizzazione di `cProfile` |

Regola #4 della pipeline: niente ottimizzazione senza profiling. Questi due sono lo strumento
per rispettarla quando si arriva alla criticità #6.

### Cython / Rust — deliberatamente esclusi per ora

La riscrittura della generazione mosse è ipotizzata in §4.3 del piano ma **solo dopo aver
misurato**. Aggiungere `cython` o `maturin` all'ambiente adesso è un invito a ottimizzare
prematuramente. Si aggiungeranno se e quando il profiler lo confermerà.

---

## Note su VRAM e stabilità

6 GB sono sufficienti per il modello previsto (~1,5 GB in fp16 con batch 512), ma il margine
non è enorme se il portatile usa la stessa GPU per il display.

- Limitare la potenza prima delle sessioni notturne: `nvidia-smi -pl <watt>`
- `scripts/preflight.py` controlla VRAM libera e temperatura prima di lanciare un training
- Se compare OOM: ridurre il batch a 256 e attivare `gradient_accumulation` invece di
  ridurre il modello

## Riproducibilità

`requirements.txt` contiene versioni **pinnate** (`==`). Le versioni minime flessibili vanno
bene per una libreria; qui l'obiettivo è che un run di agosto sia riproducibile a novembre.

Il commit hash e la config completa vengono scritti dentro ogni checkpoint
(regola #2 della pipeline), quindi un checkpoint è sempre ricollegabile all'ambiente che l'ha prodotto.
