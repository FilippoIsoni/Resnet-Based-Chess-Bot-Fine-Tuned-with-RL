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
