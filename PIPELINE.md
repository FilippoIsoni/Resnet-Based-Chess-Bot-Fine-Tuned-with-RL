# Pipeline di lavoro

Documento operativo. Il *cosa* costruire sta in [piano-motore-scacchi.md](piano-motore-scacchi.md);
qui c'è il **come si lavora**: ordine dei passi, gate di verifica, criteri di avanzamento.

**Principio guida:** in questo progetto i bug non crashano. Un encoder sbagliato, un segno
invertito nel backup MCTS o un leakage nello split non producono nessuna eccezione — producono
un motore mediocre che scopri due settimane dopo. Di conseguenza la pipeline è costruita attorno
alla verifica, non attorno alla scrittura di codice.

---

## 0. Regola fondamentale: nessuno stadio avanza senza il suo gate

Ogni stadio della pipeline ha un **gate**: un insieme di controlli automatici che devono passare
prima di poter iniziare lo stadio successivo. Il gate non è una formalità e non si "salta per ora".

```
stadio N  →  [gate N verde]  →  stadio N+1
```

Se un gate è rosso, le uniche mosse legali sono: correggere, oppure documentare in
[docs/DECISIONS.md](docs/DECISIONS.md) perché il criterio va cambiato. Mai proseguire e basta.

I gate si lanciano con:

```powershell
python scripts/check.py --stage <nome>     # un singolo gate
python scripts/check.py --all              # tutti i gate applicabili allo stato attuale
```

---

## 1. I tre livelli di check

| Livello | Quando gira | Durata | Comando |
|---|---|---|---|
| **L1 — pre-commit** | ad ogni commit, automatico | < 20 s | `scripts/check.py --level 1` |
| **L2 — pre-merge / fine sessione** | prima di chiudere un pezzo di lavoro | 2-5 min | `scripts/check.py --level 2` |
| **L3 — gate di fase** | prima di passare alla fase successiva | 10-60 min | `scripts/check.py --stage phaseN` |

### L1 — pre-commit (veloce, sempre)
- `ruff` (lint) e `ruff format --check` (formattazione)
- `mypy` sui moduli con annotazioni
- test unitari marcati `fast` (encoder, codifica mosse, utility)
- controllo che non entrino in git: pesi, dataset, PGN, `.env`

### L2 — pre-merge (completo)
- Tutta la suite `tests/unit` + `tests/integration`
- Test golden (§2.5 del piano): confronto con file JSON di riferimento versionati
- Property test randomizzati (round-trip mossa↔indice su posizioni casuali)
- Smoke run: training di 50 step su un dataset giocattolo, deve far scendere la loss

### L3 — gate di fase
Sono i criteri numerici elencati in §3. Sono lenti perché coinvolgono partite vere.

---

## 2. Ciclo di lavoro su ogni singolo task

Vale per qualunque pezzo di codice, non solo per le fasi grosse.

1. **Scrivi prima il test.** Per l'encoder e la codifica mosse questo non è dogma TDD: è
   l'unico modo di sapere se funzionano, perché l'output è un tensore che a occhio non dice nulla.
2. **Implementa** il minimo che fa passare il test.
3. **Check L1.** Se rosso, si torna al punto 2.
4. **Verifica incrociata** — il check che conta davvero: confronta il risultato con una fonte
   *indipendente* dalla tua implementazione. Esempi concreti:
   - encoder → ricostruisci la FEN dal tensore e confrontala con `board.fen()`
   - codifica mosse → round-trip su tutte le mosse legali di 10.000 posizioni
   - baseline minimax → confronto della valutazione con conteggio materiale calcolato a mano
   - MCTS → posizioni con matto forzato noto
5. **Commit** con messaggio che dice cosa è stato verificato, non solo cosa è stato scritto.
6. **Aggiorna** [docs/JOURNAL.md](docs/JOURNAL.md) se hai imparato qualcosa di non ovvio.

### Il commit va fatto solo con gate L1 verde
L'hook pre-commit lo impone. Bypassarlo con `--no-verify` è una decisione da annotare, non
un'abitudine.

---

## 3. Gli stadi e i loro gate

### Stadio 0 — Setup
Ambiente riproducibile, dipendenze pinnate, hook installati.

**Gate 0** (`--stage setup`)
- [ ] venv creato e attivo, `python -c "import torch; assert torch.cuda.is_available()"`
- [ ] `torch.cuda.get_device_name(0)` riporta la 3050
- [ ] `pytest tests/` gira (anche con zero test) senza errori di import
- [ ] hook pre-commit installato e funzionante (test: commit fittizio con file lintato male → rifiutato)
- [ ] `configs/default.yaml` caricabile e validato dallo schema
- [ ] seed globale riproducibile: due run dello stesso script danno lo stesso output

### Stadio 0-bis — Compatibilità macOS `[assegnato: collaboratore su Mac]`

**Prossimo passo per chi lavora su Mac.** Va fatto *prima* dello Stadio 1: se l'ambiente
non è portabile, ogni modulo scritto nel frattempo va riverificato due volte.

Il progetto è stato impostato su Windows + RTX 3050. Su Mac cambia il backend di calcolo:
niente CUDA, quindi il device è **MPS** (Apple Silicon) o CPU (Intel). Diversi criteri del
Gate 0 non si applicano e vanno resi condizionali invece che rimossi.

**Gate 0-bis** (`--stage setup` su macOS)
- [ ] `py -3.11` → su Mac è `python3.11`; venv creato e attivo
- [ ] **torch senza indice CUDA**: su macOS `pip install torch` prende la build giusta,
      l'`--index-url .../cu126` non ha wheel per Mac e fallisce. Documentare il comando corretto
- [ ] `torch.backends.mps.is_available()` su Apple Silicon, oppure CPU dichiarata esplicitamente
- [ ] `scripts/check.py --stage setup` verde: i check `torch + CUDA` e `VRAM >= 5 GB` devono
      diventare **SKIP con motivazione**, non FAIL — un criterio che non si applica non è un
      criterio fallito
- [ ] `scripts/preflight.py` gira senza `nvidia-smi`: sostituire con l'equivalente Mac
      (`powermetrics`, o SKIP dichiarato) e `pmset -g batt` al posto del check WMI su Win32_Battery
- [ ] `python scripts/check.py --level 2` verde
- [ ] Stockfish: binario macOS (arm64 o x86_64) in `tools/stockfish/`, percorso in config
- [ ] Tutti i pin di `requirements*.txt` risolvibili su macOS; se una versione non ha wheel
      Mac, annotarlo in `docs/DECISIONS.md` invece di allentare il pin di nascosto

**Cosa NON va cambiato:** i criteri numerici dei gate successivi. Il device cambia, la
correttezza no — l'encoder deve produrre gli stessi tensori bit a bit su entrambe le macchine.

> **Verifica incrociata fra le due macchine** — il vero motivo per cui questo stadio esiste:
> una volta pronto lo Stadio 1, i file golden in `tests/golden/` devono passare **identici**
> su Windows e su Mac. Se divergono, c'è una dipendenza dalla piattaforma dentro l'encoder
> (ordine di iterazione, dtype di default, endianness nello storage) — esattamente il tipo di
> bug silenzioso contro cui è costruita questa pipeline. Due macchine diverse sono un test
> che una sola non può fare.

**Nota sul training:** MPS regge la Fase 3 ma è più lento di CUDA e alcune operazioni fanno
fallback su CPU. La divisione del lavoro sensata è: sviluppo e test su entrambe, training
pesante (Fasi 3 e 5) sulla macchina con la 3050.

### Stadio 1 — Encoder e codifica mosse (§2.3-2.4)
Il fondamento. Tutto il resto ci si appoggia, quindi qui i check sono più severi che altrove.

**Gate 1** (`--stage encoding`) — la batteria di §2.5, tutta obbligatoria
- [ ] 20 posizioni note → tensore atteso da `tests/golden/positions.json`
- [ ] round-trip `move → indice → move` su ≥ 10.000 mosse legali, 0 fallimenti
- [ ] simmetria specchiata: posizione bianca vs stessa a colori invertiti e specchiata →
      tensori identici **e indice mossa correttamente specchiato** (criticità #1)
- [ ] promozione a donna codificata nei piani "mossa da regina", non nelle sottopromozioni (criticità #4)
- [ ] somma dei piani 0-11 == numero di pezzi sulla scacchiera, su 10.000 posizioni casuali
- [ ] encoder vs re-parsing su 1000 posizioni: nessuna deriva
- [ ] nessun indice generato fuori da [0, 4671]

### Stadio 2 — Baseline minimax (Fase 1)
**Gate 2** (`--stage baseline`)
- [ ] test di perft su posizioni standard (verifica indirettamente la generazione mosse)
- [ ] valutazione simmetrica: `eval(pos) == -eval(pos a colori invertiti)` (criticità #2, prima occorrenza)
- [ ] trova matto in 1 in 50/50 posizioni di test
- [ ] batte un giocatore random 100/100
- [ ] misura Elo vs Stockfish `UCI_Elo=1400`: risultato registrato in `docs/RESULTS.md`

### Stadio 3 — Pipeline dati (Fase 2)
Lo stadio più fragile del progetto. Il gate qui è quello che ripaga di più.

**Gate 3** (`--stage data`)
- [ ] **split per PARTITA**: test che verifica 0 game-id condivisi fra train/val/test (criticità #3)
- [ ] **0 FEN condivise** fra gli split dopo la deduplica
- [ ] statistiche del dataset stampate e registrate: n. posizioni, distribuzione esiti,
      distribuzione Elo, % con `[%eval]`, top-20 FEN per frequenza
- [ ] capping aperture verificato: nessuna FEN oltre la soglia configurata
- [ ] **ispezione manuale di 20 campioni estratti a caso**: FEN ricostruita + mossa decodificata,
      rigiocate in `python-chess` → mossa legale, posizione coerente. Nessun gate automatico
      sostituisce questo controllo, va fatto con gli occhi almeno una volta.
- [ ] round-trip storage: scrivi shard → rileggi → i tensori combaciano bit a bit
- [ ] distribuzione degli esiti sensata (né 90% patte né 90% vittorie bianco)

### Stadio 4 — Training supervisionato (Fase 3)
**Gate 4** (`--stage train`)
- [ ] **overfit test**: la rete deve raggiungere ~100% di accuracy su 512 posizioni fisse.
      Se non ci riesce c'è un bug, non un problema di capacità. Da fare *prima* del training vero.
- [ ] mascheratura delle mosse illegali identica fra training e inferenza (test dedicato)
- [ ] checkpoint: salva → ricarica → la loss riprende dallo stesso valore (± epsilon)
- [ ] resume dopo interruzione forzata: riparte dallo stesso step, stesso ordine dati
- [ ] top-1 accuracy validation in 45-52% (sotto: sospetta criticità #1)
- [ ] loss del valore ~0.45, non in divergenza
- [ ] curve loggate su TensorBoard e screenshot in `docs/RESULTS.md`

### Stadio 5 — MCTS (Fase 4)
**Gate 5** (`--stage mcts`) — batteria §4.5
- [ ] matti forzati in 1 trovati sempre, anche a 50 simulazioni
- [ ] matti in 2-3 trovati a 400-800 simulazioni
- [ ] rete casuale + MCTS batte nettamente rete casuale sola
- [ ] simmetria colori nella valutazione (criticità #2, verifica definitiva)
- [ ] **forza monotonicamente crescente** al crescere delle simulazioni (50→200→800):
      il test più diagnostico dell'intero progetto
- [ ] **diagnosi §4.4**: match 200 partite policy pura vs policy+MCTS 400 sim.
      Guadagno < 150 Elo → applicare le mitigazioni prima di procedere alla Fase 5
- [ ] profiling registrato: quota di tempo in generazione mosse vs forward della rete (criticità #6)

### Stadio 6 — Reinforcement Learning (Fase 5)
**Gate 6 di ingresso** (`--stage rl-entry`) — da superare *prima* della prima iterazione
- [ ] batching fra partite implementato e misurato: ≥ 5× rispetto a partite sequenziali (criticità #8)
- [ ] aperture randomizzate da libro bilanciato attive (criticità #7)
- [ ] gating a 200 partite (o SPRT) implementato e testato su due reti note-diverse
- [ ] criteri di stop scritti in `configs/rl.yaml`, non "a sentimento" (criticità #9)
- [ ] checkpoint per iterazione: buffer + pesi + RNG state, resume testato

**Gate 6 per ogni iterazione** — automatico, ad ogni ciclo
- [ ] % patte < 70% (sopra: aumentare diversità aperture)
- [ ] gating superato (≥ 55% su 200 partite) → promuovi; altrimenti → scarta e logga
- [ ] Elo cumulativo e intervallo di confidenza aggiornati in `docs/RESULTS.md`
- [ ] stop se: gating mai passato per 10 iterazioni / < 50 Elo dopo 25 / 40 iterazioni totali

### Stadio 7 — Deployment web (Appendice B) `[diviso fra due persone]`

Il motore è forte 1964 ± 62 Elo ma raggiungibile solo da terminale, da chi ha repo, venv e
pesi. Questo stadio lo rende giocabile da un browser.

**Il vincolo che decide l'architettura:** GitHub Pages serve solo file statici, non esegue
Python. Quindi la UI compilata sta su Pages e il motore gira altrove come servizio HTTP.

| Metà | Assegnata a | Cosa | Dove |
|---|---|---|---|
| **Frontend** | Filippo (Windows) | UI Flutter Web, deploy su GitHub Pages | `web/`, `.github/workflows/pages.yml` |
| **Backend** | collaboratore su Mac | server FastAPI, inferenza torch, deploy su HF Spaces | `src/chessbot/api/`, `deploy/`, `requirements-serve.txt` |

**`docs/API_CONTRACT.md` è il contratto vincolante fra le due metà.** Chi ha bisogno di
cambiare forma delle richieste, codici d'errore, origini CORS o la convenzione di segno di
`eval` modifica prima quel file e lo comunica all'altro. Cambiare il proprio lato sperando
che l'altro se ne accorga è il modo garantito di perdere una serata sul collegamento.

**Nessuna delle due metà blocca l'altra.** La UI si costruisce contro un `FakeEngine` che
restituisce una mossa legale casuale, quindi può essere finita e pubblicata prima che il
backend esista; il backend si verifica con `curl` e pytest. Il collegamento finale è un
cambio di URL.

**Gate 7-frontend** (verifica manuale, non c'è `--stage` per Flutter)
- [ ] `flutter analyze` senza segnalazioni e `flutter test` verde
- [ ] partita intera giocabile contro `FakeEngine`: mosse legali, arrocco, promozione, en passant
- [ ] esito corretto su matto, stallo, materiale insufficiente, ripetizione
- [ ] la barra di valutazione **non si inverte ad ogni mossa** — l'API espone `eval` dal punto
      di vista di chi muove, la barra lo vuole dal punto di vista del bianco (criticità #2
      estesa alla UI: attraversa tre confini e nessuno crasha se sbagliata)
- [ ] `flutter build web --release --base-href /<nome-repo>/` completa
- [ ] la pagina pubblicata non è bianca — senza `--base-href` corretto lo è, e la console non
      dice niente di utile
- [ ] stati di errore visibili all'utente: backend irraggiungibile, 429, 503, timeout

**Gate 7-backend** (`--stage api`, da implementare in `scripts/check.py`)
- [ ] `src/chessbot/api/` implementato oltre `__init__.py`
- [ ] `tests/unit/test_api.py` verde, con evaluator finto — i test `fast` non caricano 48 MB di pesi
- [ ] test sul **segno di `eval`**: dato un evaluator a valore noto, la risposta ha il segno
      del lato che muove
- [ ] FEN malformato o illegale → 422, mai 500
- [ ] posizione terminale → `move: null` senza eccezioni
- [ ] `/health` risponde **mentre** è in corso una ricerca a 800 simulazioni — se si blocca,
      l'handler è stato scritto `async def` e la ricerca CPU-bound sta bloccando l'event loop
- [ ] CORS: header presente per l'origine di Pages, assente per un'origine non autorizzata
- [ ] `requirements-serve.txt` coerente con la scelta torch (niente `onnxruntime`)
- [ ] tempi misurati sullo Space reale per i tre livelli; se `hard` supera ~8 s si scende a
      400 simulazioni — decisione sui dati, non sulla stima

---

## 4. Regole permanenti

Valgono per tutta la durata del progetto.

1. **Ogni numero riportato ha un intervallo di confidenza.** Con 200 partite l'errore su una
   misura Elo è ±35. Una differenza di 20 Elo non è una differenza.
2. **Ogni run è riproducibile**: seed fissato, config salvata accanto ai pesi, commit hash
   scritto nel checkpoint. Un risultato che non sai riprodurre non è un risultato.
3. **Ogni esperimento va registrato** in `docs/RESULTS.md` anche — soprattutto — se fallito.
4. **Niente ottimizzazione senza profiling.** Vale in particolare per la riscrittura in
   Cython/Rust della generazione mosse: prima si misura (criticità #6).
5. **I dati grandi non entrano in git.** `data/`, `runs/`, `*.pt`, `*.pgn` sono in `.gitignore`.
   Per i pesi: release GitHub o storage esterno.
6. **Un check che non è automatizzato non esiste.** Se scopri un bug, il primo commit del fix
   contiene il test che lo avrebbe intercettato.
7. **Prima di ogni sessione lunga di training**: `scripts/preflight.py` — spazio disco, VRAM
   libera, temperatura GPU, power limit, checkpoint dir scrivibile.

---

## 5. Struttura del repository

```
chess-bot/
├── src/chessbot/
│   ├── encoding/     # tensore 19×8×8, indici mossa 0-4671   ← Stadio 1
│   ├── data/         # parsing PGN, filtri, split, storage    ← Stadio 3
│   ├── model/        # ResNet, policy head, value head WDL    ← Stadio 4
│   ├── search/       # MCTS PUCT, batching, libro aperture    ← Stadio 5
│   ├── training/     # loop supervisionato + Expert Iteration ← Stadi 4, 6
│   ├── eval/         # match, Elo, SPRT, suite tattiche
│   ├── baseline/     # minimax + PST                          ← Stadio 2
│   ├── api/          # FastAPI per il deployment              ← Stadio 7 [backend]
│   └── utils/        # config, seed, logging, checkpoint
├── web/              # app Flutter: UI su GitHub Pages        ← Stadio 7 [frontend]
│   ├── lib/          #   engine/ (interfaccia + client HTTP + finto), game/ (UI)
│   └── test/         #   test Dart, girano con `flutter test`
├── deploy/           # Dockerfile e istruzioni per HF Spaces  ← Stadio 7 [backend]
├── tests/
│   ├── unit/         # veloci, girano ad ogni commit
│   ├── integration/  # lenti, girano nei gate
│   └── golden/       # file JSON di riferimento, versionati
├── scripts/          # check.py, preflight.py, entrypoint delle fasi
├── configs/          # YAML: default, train, rl, eval
├── docs/             # JOURNAL, RESULTS, DECISIONS, API_CONTRACT
├── data/             # ignorato da git
└── runs/             # checkpoint e log, ignorato da git
```

`web/` sta fuori da `src/` di proposito: `src/` è il package Python (`pyproject.toml` ha
`where = ["src"]`) e un progetto Dart lì dentro confonderebbe sia pip sia gli strumenti.

---

## 6. Ordine di attacco

Segue §"Ordine di attacco" del piano, con i gate innestati:

| # | Cosa | Gate da superare |
|---|---|---|
| 1 | Setup ambiente | Gate 0 |
| 1-bis | Compatibilità macOS `[collaboratore su Mac]` | Gate 0-bis |
| 2 | Encoder + codifica mosse + batteria test | **Gate 1** |
| 3 | Baseline minimax | Gate 2 |
| 4 | Pipeline dati | **Gate 3** |
| 5 | Training supervisionato | Gate 4 |
| 6 | MCTS + libro + diagnosi §4.4 | **Gate 5** → bot funzionante |
| 7 | Expert Iteration | Gate 6 |
| 8a | UI Flutter + GitHub Pages `[Filippo]` | Gate 7-frontend |
| 8b | Backend FastAPI + HF Spaces `[collaboratore su Mac]` | Gate 7-backend |

I gate in grassetto sono quelli che proteggono dalle criticità silenziose. Sono i meno
gratificanti da scrivere e gli unici che salvano settimane.

Gli ultimi due sono paralleli e indipendenti: 8a si sviluppa contro un motore finto, 8b si
verifica con `curl`. Si incontrano solo alla fine, e il punto d'incontro è
`docs/API_CONTRACT.md`.
