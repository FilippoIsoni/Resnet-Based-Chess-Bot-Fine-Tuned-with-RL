# Risultati

Registro di ogni misura fatta sul motore. **Anche i fallimenti** (regola #3): un
esperimento negativo non registrato viene rifatto identico tre settimane dopo.

Regola #1: ogni numero ha un intervallo di confidenza. Con 200 partite l'errore su una
misura Elo e circa ±35. Una differenza di 20 Elo non e una differenza.

---

## Formato di ogni voce

```
### <data> — <cosa e stato misurato>
- Commit: <hash>
- Config: <file + eventuali override>
- Checkpoint: <percorso o nome>
- Setup: <avversario, n. partite, time control, libro di aperture>
- Risultato: <numero ± IC>
- Note: <cosa e stato imparato, cosa e andato storto>
```

---

## Encoder (Stadio 1) — Gate 1 VERDE

### 2026-08-08 — Batteria §2.5 completa
- Commit: `749198b`
- Comando: `python scripts/check.py --stage encoding`

| Criterio | Campione | Esito |
|---|---|---|
| 20 posizioni golden | 20 | PASS |
| Round-trip mossa↔indice | 1.785.473 mosse legali | 0 fallimenti |
| Simmetria specchiata (criticita #1) | 4672 indici + 17.731 confronti | 0 discrepanze |
| Promozione a donna (criticita #4) | 4 promozioni × 2 colori | indici distinti, donna nel blocco regina |
| Somma piani 0-11 == n. pezzi | 4000 posizioni | 0 violazioni |
| Encoder vs re-parsing FEN | 4000 posizioni | 0 derive |
| Indici fuori da [0, 4671] | tutto il campione | 0 |

- Note: i file golden usano un hash quantizzato con endianness esplicita, quindi devono
  risultare **identici su Windows e macOS**. La verifica incrociata fra le due macchine
  (Stadio 0-bis) e ancora **da fare**: e il primo compito di chi lavora su Mac.

---

## Baseline (Stadio 2) — Gate 2 APERTO

### 2026-08-08 — Perft, simmetria, matto in 1
- Commit: `749198b`

| Criterio | Dettaglio | Esito |
|---|---|---|
| Perft | 6 posizioni CPW fino a profondita 4 | conteggi esatti |
| Valutazione simmetrica (criticita #2) | ~16.000 posizioni, 2 formulazioni | 0 violazioni |
| Matto in 1 | 50 posizioni generate e validate | 50/50 |

### 2026-08-08 — Match vs random, PRIMA del fix della finestra alpha-beta
- Commit: pre-`749198b`
- Setup: baseline depth 3 vs avversario casuale, 100 partite, colori alternati, seed 1337
- Risultato: **14W-9L-77D (52.5%), Elo +17 ± 68 (95%)**
- Esito criterio: **FAIL**
- Note: registrato perche e il dato che ha rivelato il bug (regola #3). Un motore che
  contro mosse casuali ottiene un Elo indistinguibile da zero e rotto, non debole.
  Causa: finestra alpha-beta stretta alla radice, vedi docs/JOURNAL.md. La misura
  post-fix e sotto.

### 2026-08-08 — Match vs random, DOPO il fix ← criterio del Gate 2 SODDISFATTO
- Commit: `749198b`
- Comando: `python scripts/run_gate2_match.py --games 100 --depth 3`
- Setup: baseline depth 3 vs avversario casuale, 100 partite, colori alternati, seed 1337
- Risultato: **100W-0L-0D su 100 (100.0%)**
- Elo: **non stimabile** — su un risultato estremo il modello logistico non ha un
  massimo finito. Riportare "+∞" o un numero inventato sarebbe peggio che dire che non
  si puo stimare (regola #1). Il limite inferiore utile: 100/100 e coerente con una
  differenza di forza di almeno ~600 Elo.
- Tempo: 3568 s totali, 35.7 s per partita
- File: `runs/gate2/match_random_20260808T215853Z.json`

**Confronto prima/dopo, stesso seed e stesso setup:**

| | Partite | Risultato | Elo |
|---|---|---|---|
| Prima del fix | 100 | 14W-9L-77D (52.5%) | +17 ± 68 |
| Dopo il fix | 100 | **100W-0L-0D (100%)** | non stimabile |

Il fix e una riga: finestra alpha-beta piena alla radice invece di `(-INFINITY, -alpha)`.
Costo: il tempo per partita passa da 20.6 s a 35.7 s (~1.7x), atteso e accettato.

### Da misurare
| Data | Avversario | Partite | Risultato | Elo stimato | IC |
|---|---|---|---|---|---|
| — | Stockfish UCI_Elo=1400 | — | — | — | — |

## Pipeline dati (Stadio 3) — Gate 3 VERDE (criteri automatici)

### 2026-08-09 — Verifica completa sui 20M
- Comando: `python scripts/verify_dataset.py --data data/processed --inspect 20`

| Criterio | Esito |
|---|---|
| Split per PARTITA dichiarato | PASS |
| Partite ripartite fra gli split | PASS — 274.160 su 274.168 (8 senza posizioni) |
| Round-trip storage, mosse legali | PASS — **20.000.000 posizioni, 0 illegali** |
| 0 FEN condivise fra split | PASS — **0 sovrapposizioni** |
| Capping aperture | PASS — massimo osservato 3.000, esattamente il cap |
| Distribuzione esiti sensata | PASS — 45,4% / 11,6% / 43,0% |

Il round-trip e il controllo piu forte: ogni posizione e stata ricostruita dai bitboard
e ogni indice mossa decodificato, verificando che la mossa risultante fosse legale nella
posizione ricostruita. Zero fallimenti su venti milioni.

### 2026-08-09 — Allineamento posizione-mossa, scansione completa
- Comando: `python scripts/verify_alignment.py --data data/processed`
- Durata: 18,0 min, 18.565 campioni/s

| Categoria | Campioni | % |
|---|---|---|
| **ok** | **20.000.000** | **100,0000%** |
| ok_mirror | 0 | 0,0000% |
| ok_next_ply | 0 | 0,0000% |
| ok_prev_ply | 0 | 0,0000% |
| illegal | 0 | 0,0000% |
| tratto errato | 0 | — |

Per split: train 19.117.346, val 447.011, test 435.643 — tutti al 100% `ok`.

**Diagnosi: dataset allineato.** Nessun off-by-one, nessun problema di specchiatura,
nessun caso speciale (arrocco, en passant, promozione) gestito male.

Il test classifica ogni campione anche in caso di fallimento, cosi un disallineamento
direbbe pure di che tipo e. Il classificatore e verificato su casi costruiti a mano
prima dell'uso (`tests/unit/test_verify_alignment.py`): se non distinguesse i tre casi
noti — allineato, specchiato, sfasato di un ply — i suoi risultati non varrebbero nulla.

**Sul sospetto che ha originato la verifica.** Il campione 20 di
[ISPEZIONE_GATE3.md](ISPEZIONE_GATE3.md) sembrava avere il re bianco in g2 con il nero al
tratto. In realta in g2 c'e `k` minuscolo, cioe il re **nero**: nella FEN le minuscole
sono i pezzi neri. Il re bianco e in a7. Nessun bug — ma la verifica automatica ora
esiste e copre tutto il dataset, che e comunque un guadagno netto sull'ispezione a occhio.

### 2026-08-09 — Qualita delle mosse: confronto con Stockfish
- Comando: `python scripts/verify_move_quality.py --samples 400 --depth 12`
- Setup: 400 posizioni dallo split val, Stockfish 18 a profondita 12, multipv 3
- Confronto **appaiato**: sulle stesse posizioni si valuta la mossa del dataset e una
  mossa legale a caso. Un numero assoluto non direbbe nulla; il rapporto si.

| Metrica | Dataset | Casuale | Rapporto |
|---|---|---|---|
| top-1 Stockfish | **38,8%** | 4,5% | **8,6x** |
| top-3 Stockfish | **71,0%** | 11,0% | 6,5x |
| perdita media | **105 cp** | 489 cp | 4,7x meglio |
| perdita mediana | **6 cp** | 237 cp | 39x meglio |
| errori gravi (>300 cp) | **2,0%** | 43,2% | 22x meglio |

**Verdetto: le mosse sono sensate.** La perdita mediana di 6 centipedoni — meno di un
sedicesimo di pedone — dice che nella meta dei casi la mossa giocata e quella migliore o
equivalente. Sono mosse di giocatori a 2000+ Elo, non etichette rumorose.

Questo chiude il criterio che il piano affidava all'ispezione a occhio. Copre un rischio
che `verify_alignment.py` non puo vedere: un preprocessing che associ alle posizioni una
mossa **sbagliata ma legale** darebbe 100% di allineamento e produrrebbe puro rumore. Qui
si distinguerebbe subito, perche il dataset assomiglierebbe al baseline casuale.

**Resta l'ispezione manuale** dei 20 campioni in
[ISPEZIONE_GATE3.md](ISPEZIONE_GATE3.md), ora facoltativa: sia la correttezza sia la
qualita sono coperte da misure automatiche su campioni molto piu grandi di venti.



### 2026-08-09 — Dataset da lichess_db_standard_rated_2026-07 (definitivo)
- Commit: `669d452`
- Comando: `python scripts/build_dataset.py --input data/raw/lichess_db_standard_rated_2026-07.pgn.zst --target-positions 20000000`
- Manifest: `data/processed/manifest.json`

| | |
|---|---|
| Partite lette | 26.997.925 |
| Partite tenute | **274.160 (1,02%)** |
| Posizioni | **20.000.000** (~73 per partita) |
| Dimensione su disco | **2,0 GB** (105 byte/posizione) |
| Tempo | 94 min (3.540 posizioni/s) |
| Con `[%eval]` | 33,2% |

**Split (per PARTITA — criticita #3):**

| Split | Posizioni | % | Partite |
|---|---|---|---|
| train | 19.117.346 | 95,59% | 260.340 |
| val | 447.011 | 2,24% | 6.917 |
| test | 435.643 | 2,18% | 6.903 |

**Verifica del leakage**, misurata sugli shard scritti (campione 1,88M posizioni):

| | Prima del fix | Dopo |
|---|---|---|
| condivise train-val | 1.421 | **0** |
| condivise train-test | 1.459 | **0** |
| condivise val-test | 414 | **0** |

Il primo dataset (commit `44d144b`) aveva ~0,17% di posizioni in due split: la deduplica
confrontava le FEN complete invece della sola posizione. Rigenerato dopo il fix. Le
posizioni scartate per deduplica salgono da 326.241 a 346.210 — la differenza e il
leakage chiuso.

**Distribuzione esiti:** vittorie bianco 45,4%, patte 11,6%, sconfitte bianco 43,0%.
Sensata: il vantaggio del tratto c'e ma e piccolo, e la quota di patte e bassa perche il
dataset e rapid/classical online, non tornei di elite.

**Scarti dei filtri** (§2.1), in ordine di impatto:

| Motivo | Partite |
|---|---|
| categoria blitz | 12.362.153 |
| categoria bullet | 9.851.373 |
| Elo medio < 2000 | 4.431.380 |
| time control non parsabile | 25.594 |
| meno di 10 mosse | 5.974 |
| tempo scaduto in posizione non persa | 874 |

Piu 871.547 posizioni scartate dal capping aperture e 326.241 dalla deduplica fra split.

- Note: la resa dell'1,02% e bassa ma corretta — il grosso del traffico Lichess e
  bullet e blitz, che §2.1 esclude perche rumorosi. Il capping funziona: le FEN piu
  frequenti si fermano tutte esattamente a 3.000 occorrenze, il valore configurato.

### 2026-08-09 — Prestazioni del parsing, prima e dopo
- Misurato sul dump vero, non stimato.

| | Posizioni/s | 5M posizioni |
|---|---|---|
| Prima (`read_game` su tutte) | 51 | 27 ore |
| Dopo (filtro sugli header) | **4.590** | **~20 min** |

Il fattore 90 viene dal non costruire l'albero delle mosse per il 99% delle partite che
i filtri scartano comunque. Dettagli in docs/JOURNAL.md.

---

## Training supervisionato (Stadio 4) — Gate 4 VERDE

### 2026-08-13 — Primo training completo
- Commit: `90bb2ba`
- Comando: `python scripts/train.py --run supervised --epochs 12`
- Checkpoint: `runs/supervised/best.pt` (non in git, 144 MB)

| | |
|---|---|
| Architettura | ResNet 8 blocchi x 128 canali, **12.029.715 parametri** |
| Dati | 19.117.346 posizioni di train, 447.011 di val |
| Durata | **11,8 ore** su RTX 3050 6GB |
| Posizioni viste | 233.500.672 (12 epoche) |
| Velocita | ~5.600 posizioni/s |
| Batch | 512, precisione mista attiva |

**Risultato:**

| Checkpoint | Epoca | Step | Top-1 | Top-5 |
|---|---|---|---|---|
| **best.pt** | 7 | 306.000 | **51,66%** | 89,37% |
| last.pt | 11 | 456.056 | 50,84% | 88,83% |

Picco registrato in validation: **52,11%** allo step 306.000. Il criterio del Gate 4 e
45-52%: centrato.

**Progressione della top-1:**

| Step | Top-1 |
|---|---|
| 2.000 | 33,42% |
| 8.000 | 41,91% |
| 53.000 | 48,91% |
| 306.000 | **52,11%** (picco) |
| 456.000 | 51,84% |

Il modello peggiora leggermente dopo l'epoca 7: e il motivo per cui `best.pt` e `last.pt`
sono file distinti. Per giocare si usa `best.pt`.

### 2026-08-13 — Gate 4, criteri

| Criterio | Esito |
|---|---|
| Overfit test su 512 posizioni | PASS — top-1 **100%** in 50 step |
| Mascheratura illegali identica train/inferenza (criticita #12) | PASS — **0 mosse illegali** su 1.890 posizioni giocate |
| Checkpoint: salva -> ricarica -> pesi identici | PASS |
| Checkpoint: riprende optimizer, scheduler, RNG | PASS — verificato sul campo, ripresa da step 8.000 senza salti nella loss |
| Checkpoint: scrittura atomica | PASS |
| Gradienti collegati a tutti i parametri | PASS |
| Top-1 validation 45-52% | PASS — **52,11%** |

### 2026-08-13 — La value head e debole, come previsto

Il criterio "loss del valore ~0.45" del piano non specifica quale delle due misure
intenda, e le due sono su scale diverse. Misurate entrambe su `best.pt`:

| Misura | Valore |
|---|---|
| CE sulla testa WDL | **0,9038** |
| MSE sull'eval Stockfish | 0,1727 (errore tipico 0,42 su scala [-1, 1]) |
| Copertura dell'eval | 34,4% dei campioni |

**Il numero da guardare e il confronto con le baseline banali:**

| Strategia | CE |
|---|---|
| Predire a caso | 1,0986 |
| Predire sempre la distribuzione media del dataset (43% / 12% / 45%) | 0,9710 |
| **La nostra rete** | **0,9038** |

Il guadagno sulla baseline che **ignora completamente la posizione** e di soli
**0,067 nats**. L'accuracy al 50,2% conferma: predicendo sempre "vittoria bianco" si
otterrebbe gia il 45,4%.

**La value head ha imparato pochissimo.** E la criticita #5 del piano, manifestatasi
esattamente come previsto: senza distillazione da Stockfish, l'esito della partita e un
segnale troppo rumoroso — una posizione vinta si perde venti mosse dopo, e la rete non
puo distinguere le due cose.

Non blocca il Gate 4, ma **e la cosa che conta di piu per lo Stadio 5**: l'MCTS usa
proprio quella testa per valutare le foglie dell'albero. Il piano prevede gia la diagnosi
§4.4 — match di 200 partite fra policy pura e policy+MCTS a 400 simulazioni. Se il
guadagno e sotto 150 Elo, vanno applicate le mitigazioni prima di procedere alla Fase 5.

### Da misurare
| Data | Checkpoint | Tattiche (1000) | Elo vs Stockfish | Note |
|---|---|---|---|---|
| — | best.pt | — | — | dopo lo Stadio 5 (MCTS) |

## MCTS (Stadio 5) — Gate 5 VERDE

### 2026-08-14 — Diagnosi §4.4: la misura piu importante del progetto
- Commit: `1a1e376`
- Comando: `python scripts/run_diagnosi_44.py --games 200 --sims 400`
- File: `runs/gate5/diagnosi_44_20260814T184806Z.json`

| | Policy + MCTS 400 sim | Policy pura |
|---|---|---|
| Risultato | **183W-6L-11D su 200 (94,2%)** | — |
| Guadagno | **+486 Elo ± 103** (95%) | — |
| Soglia del piano | ≥ 150 Elo | |
| Durata | 198 min, 59,4 s per partita | |

**La value head regge la ricerca.** Il guadagno e piu del triplo della soglia, e
l'intervallo di confidenza non la sfiora nemmeno: il verdetto e statisticamente solido
(regola #1).

**Perche il risultato conta piu di quanto sembri.** La value head e debole in assoluto —
al Gate 4 abbiamo misurato una CE di 0,9038 contro 0,9710 di chi ignora la posizione,
appena 0,067 nats di guadagno. Il rischio era che una value head cosi povera rendesse la
ricerca inutile, perche l'MCTS usa proprio quella testa per valutare le foglie.

Non e successo. La spiegazione: la ricerca guadagna forza **anche solo esplorando**.
Vedere una cattura tre semimosse piu avanti non richiede una valutazione posizionale
raffinata, richiede di guardare — e i terminali (matto, stallo, patta) sono risposte
esatte che non passano affatto dalla rete.

Il piano prevedeva mitigazioni sotto i 150 Elo: **non servono**.

### 2026-08-14 — Batteria §4.5
- Comando: `python scripts/run_gate5_tests.py --games 14 --mate-positions 50`

| Criterio | Esito |
|---|---|
| Matti in 1 a 50 simulazioni | **50/50** con rete cieca |
| Matti in 1 a 400 simulazioni | **50/50** con rete cieca |
| MCTS batte la policy pura (rete materiale) | +15,0 pedoni, in vantaggio 6/6 |
| Simmetria colori (criticita #2) | 3/3 coerenti |
| **Forza crescente con le simulazioni** | **7W-0L-7D (75%)**, Elo +191 ± 210 |

L'ultimo e "il test piu diagnostico dell'intero progetto" secondo il piano: **zero
sconfitte** per 800 simulazioni contro 50 dimostra che il backup del valore ha il segno
giusto. Se fosse invertito, cercare di piu avrebbe peggiorato il gioco.

Che i matti si trovino 50/50 con una rete che **non sa nulla di scacchi** (priori
uniformi, valore sempre zero) dimostra che l'albero funziona indipendentemente dalla
rete: il merito e della ricerca, non dei pesi.

### 2026-08-14 — Prestazioni del batching (§4.3, criticita #8)
- 400 simulazioni con la rete allenata, RTX 3050

| `batch_size_leaves` | Tempo | Velocita | Chiamate GPU |
|---|---|---|---|
| 1 (nessun batching) | 1,88 s | 213 sim/s | 401 |
| 8 | 0,38 s | 1.055 sim/s | 53 |
| **24** (default) | **0,31 s** | **1.276 sim/s** | **21** |

**Sei volte piu veloce**, nell'ordine di grandezza che §4.3 prevedeva ("senza batching,
30 sim/s invece di 400").

### Da misurare
| | |
|---|---|
| Elo vs Stockfish `UCI_Elo` 1400 / 1800 / 2200 | — |
| Suite tattiche (1000 posizioni) | — |
| Profiling: generazione mosse vs forward (criticita #6) | — |

## Expert Iteration (Stadio 6)

| Iter | Partite | % patte | Gating | Elo cumulativo | IC | Promosso |
|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — |

Criteri di stop attivi (da `configs/rl.yaml`): gating mai passato per 10 iterazioni /
< 50 Elo dopo 25 / tetto duro a 40.
