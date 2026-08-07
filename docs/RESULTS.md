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

## Baseline (Stadio 2)

_Da compilare dopo il Gate 2._

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
