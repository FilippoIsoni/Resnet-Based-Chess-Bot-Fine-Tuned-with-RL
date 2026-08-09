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

## Pipeline dati (Stadio 3)

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

## Training supervisionato (Stadio 4)

_Da compilare dopo il Gate 4._

| Data | Checkpoint | Top-1 policy (val) | Loss valore | Tattiche (1000) | Note |
|---|---|---|---|---|---|
| — | — | atteso 45-52% | atteso ~0.45 | — | — |

## MCTS (Stadio 5)

Diagnosi §4.4 — la misura piu importante dell'intero progetto:

| Data | Policy pura | Policy + MCTS 400 | Guadagno | IC | Esito |
|---|---|---|---|---|---|
| — | — | — | soglia: **≥150 Elo** | — | — |

Scala delle simulazioni (deve essere monotona crescente):

| Simulazioni | 50 | 200 | 800 |
|---|---|---|---|
| Elo | — | — | — |

## Expert Iteration (Stadio 6)

| Iter | Partite | % patte | Gating | Elo cumulativo | IC | Promosso |
|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — |

Criteri di stop attivi (da `configs/rl.yaml`): gating mai passato per 10 iterazioni /
< 50 Elo dopo 25 / tetto duro a 40.
