# Piano di progetto — Motore scacchistico con rete neurale

**Obiettivo:** motore di livello medio-alto (2000-2400 Elo), addestrato su hardware consumer.
**Vincolo hardware:** RTX 3050 Laptop, 6 GB VRAM.
**Stack:** Python + PyTorch (training), FastAPI + ONNX Runtime (inferenza), Flutter (UI).

---

## Indice delle fasi

| Fase | Contenuto | Elo atteso a fine fase | Tempo |
|------|-----------|------------------------|-------|
| 0 | Setup e baseline | ~1300 | 1 giorno |
| 1 | Acquisizione dati | — | 2-3 giorni |
| 2 | Encoder e storage | — | 3-4 giorni |
| 3 | Architettura | — | 1 giorno |
| 4 | Training supervisionato | ~1800 | 1-2 settimane |
| 5 | MCTS + libro aperture | ~2200 | 1 settimana |
| 6 | Distillazione | ~2400 | 3-4 giorni |
| 7 | RL / Expert Iteration *(opzionale)* | +150-250 | 3-4 settimane |
| 8 | Valutazione | — | continuo |
| 9 | Deployment | — | 2 weekend |

**Progetto minimo funzionante: fasi 0-5.** Dalla 6 in poi è ottimizzazione.

---

## Fase 0 — Setup e baseline

### Metrica di riferimento

Fissare subito checkpoint falsificabili usando Stockfish con `UCI_LimitStrength=true` e `UCI_Elo` crescente: 1400 → 1800 → 2200. È un metro calibrato e gratuito.

### Baseline stupida

Prima di toccare le reti neurali, scrivere un minimax a profondità 3-4 con valutazione materiale classica (P=1, N=B=3, R=5, Q=9) più piece-square tables. Sono ~150 righe e gioca a ~1200-1400 Elo.

**Serve come test di sanità:** se la rete neurale non batte questa baseline, c'è un bug, non un limite di capacità del modello.

### Infrastruttura da predisporre subito

- File di configurazione unico (YAML) con tutti i percorsi e gli iperparametri
- Checkpoint ogni N step che salvano **pesi + stato optimizer + posizione nel dataset**
- Logging su TensorBoard o Weights & Biases

Il checkpointing non è opzionale: su un portatile un aggiornamento di sistema o uno shutdown termico azzerano una notte di training esattamente come una disconnessione di Colab.

---

## Fase 1 — Acquisizione e filtraggio dati

### Fonti

- **database.lichess.org** — dump mensili in PGN, decine di milioni di partite. Fonte principale.
- **ChessBench** (google-deepmind/searchless_chess) — 10M partite Lichess con *ogni mossa legale* valutata da Stockfish 16 a 50 ms. Da usare in Fase 6.
- **Lumbras Gigabase / Caissabase** — partite di maestri, se si vuole uno stile più classico.

### Filtri da applicare

| Criterio | Valore | Motivo |
|----------|--------|--------|
| Elo medio dei giocatori | ≥ 2000 | Imitare gente forte, non media |
| Time control | rapid / classical | Il bullet è rumoroso |
| Esito | escludere tempo scaduto in posizione non persa | Etichetta di valore falsata |
| Annotazioni | preferire partite con `[%eval ...]` | Target di valore molto più pulito |

Circa il 6% delle partite Lichess ha le valutazioni Stockfish già nei commenti PGN. Raccogliere 2-3 milioni di posizioni annotate è un ottimo bonus gratuito.

### Gestione delle aperture

**Le aperture si tengono.** Il problema non è che non contino — contano moltissimo — ma che sono sovrarappresentate: la posizione iniziale compare in tutte le partite, e poche centinaia di posizioni si mangerebbero una fetta sproporzionata dei gradienti.

Soluzione: **capping**, non esclusione.

- Massimo 2000-5000 occorrenze per ogni FEN distinta, **oppure**
- pesare ogni campione con `1/√(conteggio_FEN)`

Così la rete impara la teoria d'apertura senza schiacciare il mediogioco.

### Volume target

**15-20 milioni di posizioni**, cioè circa 200-250k partite. Oltre non serve con una rete da 3-5M parametri: si satura prima.

---

## Fase 2 — Encoder e storage

> **Questa è la fase più fragile del progetto.** Un bug qui non fa crashare niente: alleni su etichette sbagliate e te ne accorgi dopo giorni, guardando un'accuracy che non sale.

### Cosa fa python-chess e cosa no

**Fa:** parsing del PGN, replay delle mosse, stato della scacchiera, legalità, arrocco, en passant, promozioni, matto, ripetizione.

**Non fa:** la conversione in tensori. Non esiste `board.to_tensor()`, e non per dimenticanza — quante piani usare e con che convenzioni sono scelte di modellazione, non di scacchi.

### Encoder posizione → tensore 19×8×8

Pila di 19 piani 8×8, tutti inizializzati a zero:

| Piani | Contenuto |
|-------|-----------|
| 0-5 | Pedoni, cavalli, alfieri, torri, donne, re — **lato che muove** |
| 6-11 | Stessi sei tipi — **lato avversario** |
| 12 | Turno (piano costante) |
| 13-16 | Diritti di arrocco (4 piani costanti) |
| 17 | Colonna dell'en passant, se disponibile |
| 18 | Contatore delle 50 mosse, normalizzato |

I piani 12-18 sono **la storia della partita compressa**. Gli scacchi sono markoviani: la posizione più questi bit contengono tutto il necessario. Non serve alcun modello sequenziale, e non serve alcuna storia di posizioni precedenti.

*(Nota: AlphaZero usa 8 posizioni di storia, ma serve quasi solo per le ripetizioni. Le ablation di Leela mostrano un guadagno minimo negli scacchi. Ignorabile.)*

### Orientamento — accorgimento chiave

**Orientare sempre la scacchiera dal punto di vista di chi muove.** Se tocca al Nero, specchiare verticalmente e scambiare i colori, così la rete vede sempre "il mio esercito che sale". Dimezza lo spazio da imparare, a costo zero.

### Codifica delle mosse → indice 0-4671

Schema AlphaZero: 8×8×73 = 4672.

- 56 "mosse da regina" per casella (8 direzioni × 7 distanze)
- 8 mosse di cavallo
- 9 sottopromozioni (3 pezzi × 3 direzioni)

La mappatura `chess.Move` → indice è codice proprio. È la funzione più insidiosa del progetto.

### Test di regressione — obbligatorio

Preparare un JSON con **20 posizioni note** e l'output atteso (tensore + indice mossa). Farlo girare come test automatico ad ogni modifica dell'encoder. Servirà di nuovo in Fase 9 per validare il porting.

### Storage compatto

**Non salvare i tensori float espansi.** 20M × 19 × 64 × 4 byte = 97 GB.

Salvare la forma minima:

| Campo | Dimensione |
|-------|-----------|
| 12 bitboard `uint64` | 96 byte |
| Diritti arrocco + turno + en passant | 1 byte |
| Indice mossa giocata (0-4671) | 2 byte |
| Risultato partita | 1 byte |
| Eval Stockfish (se presente) | 2 byte |
| **Totale** | **~102 byte/posizione** |

→ **~2 GB per 20M posizioni.** Formato: shard `.npy` memory-mapped da 1M posizioni. L'espansione a 19×8×8 avviene al volo nel DataLoader.

### Prestazioni del preprocessing

python-chess parsa 500-2000 partite/s per core. Usare `multiprocessing.Pool` con 6-8 worker. Conta **4-8 ore** per la conversione completa: si lancia una notte e non ci si pensa più.

---

## Fase 3 — Architettura

### Scelta: ResNet convoluzionale a due teste

```
Input 19×8×8
  └─ Conv 3×3, 128 filtri, BN, ReLU
  └─ 8 × [blocco residuo: Conv3×3-BN-ReLU-Conv3×3-BN-(+skip)-ReLU]
       ├─ Policy head:  Conv 1×1 (32ch) → flatten → Linear → 4672 logit
       └─ Value head:   Conv 1×1 (8ch)  → Linear(128) → Linear(3) → softmax WDL
```

**~3,5M parametri.** In fp16 con batch 512 occupa ~1,5 GB di VRAM: la 3050 sta larga, si potrebbe salire a 10 blocchi × 160 filtri.

Le convoluzioni 3×3 hanno senso perché le relazioni scacchistiche sono locali e spaziali (catene di pedoni, case deboli, difese reciproche) e dopo 8 blocchi il campo recettivo copre l'intera scacchiera.

### Testa WDL invece di scalare

Tre uscite softmax (vittoria/patta/sconfitta) invece di un singolo `tanh` in [−1,1]. La cross-entropy su tre classi dà gradienti più informativi della MSE, e permette di distinguere una posizione **realmente pari** da una **estremamente sbilanciata con esito incerto** — che con lo scalare valgono entrambe 0.

### Entrambe le teste servono da subito

La value head sembra accessoria se si ragiona in termini di "prevedere la mossa", ma **senza di lei l'MCTS non può esistere**: è lei che valuta le foglie. Allenarla dalla Fase 4 evita di dover rifare tutto un mese dopo.

### Alternative valutate e scartate

- **NNUE (stile Stockfish):** più forte per watt, ma valuta soltanto (niente policy) e richiede un alpha-beta serio con quiescence, transposition table, move ordering. Molto più ingegneria, molto meno ML.
- **Transformer su 64 token:** scala benissimo (DeepMind, ~2900 Elo senza ricerca) ma con 270M parametri e miliardi di posizioni annotate. Alla nostra scala la ResNet vince per bias induttivo.
- **Pesi pretrained (Maia, reti lc0):** formato protobuf incompatibile con PyTorch, architettura vincolata, e Maia è costruita per *imitare* gli umani, blunder compresi — base strana per l'RL. La Fase 4 costa una-due notti: non conviene.

---

## Fase 4 — Training supervisionato

### Loss

```
L = CE(policy_logits, mossa_giocata)
  + 0.5 · CE(wdl, risultato_partita)
  + 0.5 · MSE(value, eval_stockfish)     # solo dove disponibile
```

**Mascherare i logit delle mosse illegali a −inf prima della softmax.** Elimina di colpo un'intera classe di errori.

### Iperparametri

| Parametro | Valore |
|-----------|--------|
| Optimizer | AdamW |
| Learning rate | 1e-3, warmup 2k step, cosine decay |
| Weight decay | 1e-4 |
| Batch size | 512 |
| Precisione | mixed (`torch.amp`) |
| Label smoothing (policy) | 0.05 |
| Epoche | 8-15 |
| DataLoader | `num_workers=6`, `pin_memory=True` |

### Tempi realistici

~2000-3500 posizioni/s in training → **1,5-2 ore per epoca** su 15M posizioni → ciclo completo in **una-due notti**.

Se la GPU sta al 40% di utilizzo, il collo di bottiglia è il DataLoader, non la rete.

### Metriche, in ordine di utilità

1. **Top-1 accuracy della policy** — target 45-52%. Non salirà oltre: gli umani non sono deterministici.
2. **Accuracy su 1000 tattiche** di Lichess Puzzles tenute da parte — il termometro più onesto.
3. **Loss del valore** — non scenderà molto sotto 0.45 con etichette da risultato finale: l'etichetta è intrinsecamente rumorosa.

### Risultato atteso

Rete che gioca "a istinto" (mossa col logit più alto, zero ricerca): **1700-1900 Elo**.

Non 2000, nonostante i dati siano di giocatori 2000: imitare la mossa *media* di una popolazione non equivale a giocare come quella popolazione. Quando due forti divergono, la rete impara una miscela sfocata, e nelle posizioni tattiche acute la sfocatura è fatale.

### Nota sull'hardware

Limitare la potenza GPU (`nvidia-smi -pl`), tenere il batch moderato, sollevare il portatile per l'aerazione. Si perde il 10-15% di velocità e si evita il thermal throttling a metà notte — su sessioni da 12 ore è un guadagno netto.

---

## Fase 5 — Ricerca MCTS e libro d'aperture

> **È il salto di Elo più grande di tutto il progetto, e non costa un solo step di training.**

### MCTS con PUCT

Ad ogni simulazione si scende nell'albero massimizzando:

```
Q(s,a) + c_puct · P(s,a) · √(Σ_b N(s,b)) / (1 + N(s,a))
```

dove `P` viene dalla policy della rete, `Q` è la media dei valori osservati, `N` sono le visite. Arrivati a una foglia la si valuta **con la rete**, e si propaga il valore indietro. Si gioca la mossa più visitata.

`c_puct` ≈ 1.5-2.5.

### Niente rollout casuali

Il Monte Carlo classico (giocare a caso fino alla fine) **fallisce disastrosamente negli scacchi**. Il Go è accumulativo: da una posizione vinta, giocare a caso finisce spesso in vittoria. Gli scacchi sono a morte improvvisa: con la donna in più, tre mosse casuali possono regalare il matto. Il rumore tattico distrugge il segnale posizionale.

La value head **è** ciò che rende praticabile la ricerca. Il "Monte Carlo" nel nome MCTS è ormai un residuo storico: quello che resta è la parte *tree search*.

### Ottimizzazioni indispensabili

- **Batching delle foglie con virtual loss** — valutare 16-32 posizioni per forward pass. Senza, la GPU resta ferma il 95% del tempo: 30 simulazioni/s invece di 400.
- Riconoscimento diretto di matto, stallo, ripetizione e regola delle 50 mosse **prima** di chiamare la rete.

### Attenzione al collo di bottiglia

python-chess in puro Python genera 20-50k mosse/s. Sarà questo — non la rete — a limitare la profondità. Se il profiler lo conferma, riscrivere la generazione mosse in Cython o Rust. **Solo dopo aver misurato.**

### Libro d'aperture

Montare un libro **Polyglot** (`.bin`) per le prime 8-12 mosse: `chess.polyglot` in python-chess, ~20 righe di codice. Costo computazionale zero, apertura di livello professionale garantita indipendentemente da cosa ha imparato la rete.

È ciò che fanno tutti i motori seri, Stockfish compreso.

### Risultato atteso

**2100-2300 Elo.** Il progetto è già funzionante e già ampiamente sufficiente all'obiettivo dichiarato.

---

## Fase 6 — Distillazione da Stockfish

> **Il miglior rapporto risultato/fatica dell'intero progetto: +300-500 Elo per una notte di CPU.**

Formalmente è apprendimento supervisionato, non RL. Ma è la cosa da fare **prima** di qualsiasi self-play.

### Procedura

1. Prendere 500k-1M posizioni (o attingere direttamente a **ChessBench**, che ha già tutto annotato)
2. Se si annota in proprio: Stockfish a profondità 12-15, multi-PV, ~0,1 s/posizione
3. Allenare la rete a riprodurre **sia la valutazione sia la distribuzione delle mosse migliori**
4. Learning rate ridotto (1e-4), poche epoche

L'annotazione è CPU-bound e imbarazzantemente parallela: candidato ideale per Colab/Kaggle mentre il PC fa altro.

### Target del valore migliorato

Passare dal risultato puro `z` a un **target misto**, come fa Leela:

```
target = α · Q_radice(s) + (1 − α) · z        con α ∈ [0.3, 0.5]
```

`Q_radice` è già disponibile: è la media dei valori accumulati dalle simulazioni MCTS appena fatte. **Costa una riga di codice.** È una stima *cercata*, quindi molto meno rumorosa della valutazione statica — concettualmente lo stesso trucco di TDLeaf(λ).

---

## Fase 7 — Reinforcement Learning (opzionale)

### Metodo: Expert Iteration (schema AlphaZero)

**Non PPO, non REINFORCE.**

Il ciclo: il modello gioca contro se stesso con MCTS; per ogni posizione si salva la distribuzione delle visite π e il risultato z; si allena la rete su quei dati; si ripete.

**La loss è identica a quella della Fase 4.** Cambia solo la provenienza delle etichette: prima gli umani di Lichess, ora l'MCTS di se stessi. Non serve scrivere un algoritmo di RL nuovo — si riusa il training loop esistente. Tutta la complessità sta nell'infrastruttura di generazione partite.

### Perché non PPO

- **Reward terminale e ternario** dopo 60-80 mosse: credit assignment quasi impossibile. L'MCTS invece dà un segnale **denso e locale** su ogni singola posizione.
- **Spazio azioni enorme e variabile** (4672 logit, ~35 legali, mascherati diversamente ad ogni posizione): l'importance sampling è numericamente scomodo.
- **L'MCTS è un operatore di miglioramento della policy garantito**: π è dimostrabilmente meglio di p, perché è p *più* 200 simulazioni di verifica. Non serve stimare la direzione di miglioramento — la ricerca la fornisce. È un vantaggio strutturale disponibile solo con un simulatore perfetto.

### Configurazione

| Parametro | Valore |
|-----------|--------|
| Partite per iterazione | 400-600 |
| Simulazioni per mossa | 150-200 |
| Rumore di Dirichlet (radice) | α = 0.3, peso 0.25 |
| Temperatura | 1.0 per le prime 15 mosse, poi ~0 |
| Replay buffer | ultime 8-10 iterazioni |
| Epoche per iterazione | 2-3 |
| Optimizer | SGD momentum 0.9 (più stabile di Adam sotto distribution shift) |
| Learning rate | **1e-4 o meno** |
| Penalità KL vs policy supervisionata | peso 0.05-0.1 |
| Gating | ≥55% su 100 partite (meglio 200, o test SPRT) |

**La penalità KL** è un'aggiunta rispetto ad AlphaZero originale: loro partivano da zero e non avevano nulla da dimenticare, noi partiamo da una rete che sa già le aperture umane e non vogliamo che le butti via inseguendo strategie degeneri di self-play.

**Il gating** non è un dettaglio: è l'unica cosa che impedisce di accorgersi dopo cinque giorni di essere peggiorati monotonamente.

### Costo onesto

8-15 ore per iterazione, 20-40 iterazioni necessarie. **Il 90% del tempo se ne va nella generazione partite**, non nel training. Ottimizzare il batching delle foglie MCTS *prima* di lanciare la prima iterazione.

Guadagno atteso: **150-250 Elo per settimane di macchina accesa**. Da fare per valore didattico e di portfolio (un ciclo di Expert Iteration implementato è un ottimo pezzo da mostrare), non per efficienza.

### Esperimento alternativo economico: TDLeaf(λ)

Invece di un'iterazione di self-play completo: 2000 partite veloci contro Stockfish limitato, aggiornando **solo la value head** con TDLeaf(λ=0.9) — bootstrap sulla valutazione della foglia della variante principale, non della radice.

Costa una frazione perché non serve un MCTS profondo, e dice se il collo di bottiglia è la valutazione o la policy. Il grafico della loss prima/dopo è materiale di analisi molto presentabile.

*Precedente storico: TD-Gammon (1992) raggiunse livello mondiale a backgammon con TD(λ); KnightCap (1999) e Giraffe (2015, livello Maestro Internazionale) usarono TDLeaf(λ) negli scacchi.*

---

## Fase 8 — Valutazione

### Match calibrati

- **Stockfish** con `UCI_LimitStrength`, Elo crescente da 1320
- 100+ partite per livello, con **aperture bilanciate** da un libro (es. Silver Suite) per ridurre la varianza
- Elo relativo: `400 · log10(vittorie/sconfitte)`, o BayesElo per rigore

### Sparring calibrato: Maia

I modelli **Maia** (CSSLab) esistono per fasce di rating da 1100 a 1900 e sbagliano *in modo umano*. Come banco di prova sono più informativi di Stockfish limitato, che sbaglia in modo artificiale.

Girano dentro lc0 con `go nodes 1` (ricerca disabilitata). Nota: sono più forti del rating nominale, perché giocano la mossa *media* di quella fascia.

### Suite tattiche

STS (Strategic Test Suite), Arasan, WAC.

### Rating reale

Esporre il motore via UCI e collegarlo a **lichess-bot**. Due giorni di partite danno un rating pubblico e verificabile — la soddisfazione finale del progetto.

---

## Fase 9 — Deployment

### Architettura

**Flutter solo UI** (web app statica) + **backend Python separato**. L'inferenza in-browser via WASM è stata valutata e scartata: 300 MFLOPs/posizione in WASM single-thread danno 7-10 posizioni/s, cioè 30 secondi per mossa.

Su CPU nativa la stessa rete fa 150-350 posizioni/s → **200 simulazioni in 1-2 secondi**. Nessun compromesso sulla dimensione della rete.

### API

```
POST /move
{ "fen": "...", "level": 3, "session": "uuid-opzionale" }
→ { "move": "e2e4", "eval": 0.34, "pv": ["e2e4","e7e5"], "ms": 1240 }

GET /health
```

- `level` mappa sul numero di simulazioni (50 / 200 / 800) → difficoltà regolabile. **Serve davvero:** se il motore stritola l'avversario 10-0, smette di essere divertente.
- `session` abilita il **riuso dell'albero MCTS** (cache LRU ~30 sessioni): ripartire dal sottoalbero già esplorato risparmia il 30-40% delle simulazioni. Forza gratuita.

### Esportare in ONNX, non spedire PyTorch

| | Dimensione |
|---|---|
| Wheel PyTorch CPU | > 200 MB |
| `onnxruntime` | ~15 MB |

Su un free tier con 512 MB di RAM è la differenza tra "funziona" e "OOM al primo utente". E per la sola inferenza ONNX Runtime è pure più veloce.

**Verificare sempre** che l'output ONNX combaci con quello PyTorch su una decina di posizioni prima di deployare.

### Hosting

| Piattaforma | Giudizio |
|-------------|----------|
| **Fly.io** | ✅ Consigliato — le macchine si fermano e si risvegliano in 2-5 s |
| Hugging Face Spaces | Gratis e generoso di RAM, ma risveglio lento |
| Render free | Cold start 40-60 s: troppo |
| Oracle Cloud always-free | Migliore in assoluto (ARM 4 core, 24 GB, non dorme) ma richiede configurazione |
| **Vercel serverless** | ❌ Sconsigliato per il backend: il carico è CPU-bound e sostenuto, si perde il riuso dell'albero, timeout stretti, CPU non garantita |

Per il **frontend** GitHub Pages e Vercel sono equivalenti. Vercel dà un dominio in root (niente `--base-href`) e preview deploy per branch; Pages vive nello stesso repo.

### Le tre trappole

1. **HTTPS obbligatorio** — GitHub Pages serve in https, un backend in http viene bloccato come *mixed content* senza errori utili in console.
2. **CORS** — autorizzare esplicitamente l'origine del frontend (`CORSMiddleware` in FastAPI). Sintomo tipico: funziona da Postman, fallisce dal browser.
3. **Cold start** — chiamare `/health` all'apertura dell'app con un indicatore "sto svegliando il motore", così i secondi si consumano prima che serva la prima mossa. Un ping esterno ogni 10 minuti (UptimeRobot) tiene sveglia l'istanza.

### Robustezza

- **Cap sulle simulazioni** lato server e rate limit per IP: l'URL è pubblico, basta uno script per saturare la CPU
- Timeout duro a 10 s per richiesta
- Validare il FEN con python-chess, rispondere 400 se illegale (un FEN malformato fa crashare il worker)
- Restituire l'`eval` e mostrare una **barra di valutazione** a lato della scacchiera: 20 righe di Flutter, l'app diventa dieci volte più interessante

### Note Flutter

- Pacchetti: **`bishop`** (regole e generazione mosse), **`squares`** (scacchiera, stesso autore)
- URL del backend via `--dart-define=API_URL=...`, così si punta a localhost in sviluppo senza toccare il codice
- Interfaccia astratta `Engine` con `Future<String> bestMove(String fen)` — permette di scambiare implementazioni senza toccare la UI

---

## Dove allenare

**In locale**, per il grosso. La ragione non è la potenza ma la **durata**: il training supervisionato dura 15-25 ore filate, e Colab gratis stacca le sessioni, taglia gli idle e nelle giornate di traffico può non dare alcuna GPU. La 3050 è modesta ma è propria e non si scollega. Il divario con una T4 su una rete 8×8 così piccola è ~2-2,5×.

**Colab / Kaggle** conviene per tre cose:

1. **Preprocessing dei PGN** — CPU-bound, parallelo, usa e getta
2. **Annotazione con Stockfish** (Fase 6) — idem, ed evita di tenere il portatile al 100% per una notte
3. **Esperimenti brevi** — tre learning rate su 1M posizioni, o 8 blocchi contro 10. Si può lanciare un esperimento lì mentre un altro gira in locale.

**Kaggle Notebooks** è preferibile a Colab gratis: 30 ore GPU/settimana con quota dichiarata, sessioni fino a 9 ore, ed esecuzione **in background** (si chiude il browser e continua).

Con checkpoint robusti si diventa agnostici e si decide caso per caso.

---

## Ordine di attacco consigliato

1. **Encoder + codificatore mosse + test di regressione** (Fase 2) — tutto il resto ci si appoggia
2. **Backend FastAPI con motore finto** che gioca a caso — gira prima ancora che la rete esista, e sblocca lo sviluppo dell'app in parallelo
3. **Baseline minimax** (Fase 0) — metro di paragone
4. **Pipeline dati completa** (Fase 1-2) — una notte di macchina
5. **Training supervisionato** (Fase 4)
6. **MCTS + libro** (Fase 5) → a questo punto il progetto è finito e funzionante
7. Distillazione, RL, rifiniture
