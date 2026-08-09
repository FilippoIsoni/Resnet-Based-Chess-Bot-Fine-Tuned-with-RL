# Ispezione manuale — Gate 3, Stadio 3

20 campioni estratti a caso dal dataset (seed 7, primo shard di ogni split), da
verificare **a occhio**. Il Gate 3 lo richiede esplicitamente:

> ispezione manuale di 20 campioni estratti a caso: FEN ricostruita + mossa decodificata,
> rigiocate in `python-chess` → mossa legale, posizione coerente. **Nessun gate automatico
> sostituisce questo controllo**, va fatto con gli occhi almeno una volta.

Le FEN sono ricostruite **dagli shard binari**, non rilette dal PGN: se il packing avesse
corrotto qualcosa, si vedrebbe qui.

## Come si verifica

Incolla la FEN su <https://lichess.org/editor>, guarda la scacchiera, gioca la mossa.

## Cosa cercare

Il controllo automatico ha gia verificato che ogni mossa sia **legale**. Quello che non
puo dire:

- **mosse assurde ripetute** → errore di orientamento. Sarebbero tutte legali e tutte
  sbagliate: e il modo in cui la criticita #1 si manifesterebbe qui.
- **tratto incoerente**: il campo dopo la scacchiera nella FEN (`w`/`b`) deve
  corrispondere al colore del pezzo che muove.
- **esiti tutti uguali**: devono essere misti. Solo "bianco vince" ovunque significa
  etichetta rotta.

## I tre campioni piu diagnostici

Se hai poco tempo, guarda questi:

| # | Perche |
|---|---|
| 15 | e un matto (`Qg1#`) — se il matto c'e davvero, pezzi e tratto sono allineati |
| 3 | apertura riconoscibile (Caro-Kann dopo 1.c4 c6) — facile capire se e sensata |
| 20 | `eval 8115` significa matto forzato nel nostro encoding; l'esito dice "bianco vince" |

---

### 1. [train]

```
r1b2rk1/pp2n1pp/2nq4/3p1p2/3Pp3/1BP5/PP1N1PPP/R2QNRK1 b - - 1 1
```

| tratto | mossa | notazione | esito partita | eval |
|---|---|---|---|---|
| nero | c8e6 | Be6 | bianco vince | -134 |

### 2. [train]

```
r1bq1rk1/pp4pp/3b4/4np2/2PpP3/2P2B1P/PPQ2PP1/R1B1KN1R b KQ - 0 1
```

| tratto | mossa | notazione | esito partita | eval |
|---|---|---|---|---|
| nero | d4d3 | d3 | nero vince | -137 |

### 3. [train]

```
rnbqkbnr/pp1ppppp/2p5/8/2P5/8/PP1PPPPP/RNBQKBNR w KQkq - 0 1
```

| tratto | mossa | notazione | esito partita | eval |
|---|---|---|---|---|
| bianco | d2d4 | d4 | nero vince | — |

### 4. [train]

```
8/8/8/1P2k1p1/4N1Kp/2P2P2/8/8 b - - 1 1
```

| tratto | mossa | notazione | esito partita | eval |
|---|---|---|---|---|
| nero | e5d5 | Kd5 | bianco vince | — |

### 5. [train]

```
rnb2rk1/p3ppbp/2p3p1/3qP3/2pP4/5N2/PP2B1PP/R1BQ1RK1 w - - 2 1
```

| tratto | mossa | notazione | esito partita | eval |
|---|---|---|---|---|
| bianco | f3g5 | Ng5 | bianco vince | — |

### 6. [train]

```
r1bq1rk1/1p2npbp/p1n1p1p1/3pP3/3P4/2NBBN2/PP3PPP/R2Q1RK1 w - - 4 1
```

| tratto | mossa | notazione | esito partita | eval |
|---|---|---|---|---|
| bianco | d1e2 | Qe2 | nero vince | 84 |

### 7. [train]

```
r1b2r2/2p1n2k/pb5p/1p1NNpp1/3Pp3/1BP4P/PP3P2/R1B2RK1 b - - 1 1
```

| tratto | mossa | notazione | esito partita | eval |
|---|---|---|---|---|
| nero | e7g6 | Ng6 | nero vince | 344 |

### 8. [val]

```
4r1k1/p4p2/2p2Bp1/2q5/5R2/3Q3P/P5P1/6K1 w - - 0 1
```

| tratto | mossa | notazione | esito partita | eval |
|---|---|---|---|---|
| bianco | g1h2 | Kh2 | nero vince | 687 |

### 9. [val]

```
r4rk1/p1p1b2n/3p2Bp/2p1P3/8/2N2RP1/PP5P/3R2K1 b - - 0 1
```

| tratto | mossa | notazione | esito partita | eval |
|---|---|---|---|---|
| nero | f8f3 | Rxf3 | nero vince | — |

### 10. [val]

```
rn2k1nr/pp3pb1/2p1p1p1/3pP2p/3P1PPq/2NB3P/PPP2Q2/R1B1K2R b KQkq - 2 1
```

| tratto | mossa | notazione | esito partita | eval |
|---|---|---|---|---|
| nero | h5g4 | hxg4 | bianco vince | — |

### 11. [val]

```
r4kr1/2q2p2/4p3/1Qbp4/p3pPPp/4P3/2PN3P/1R3RK1 w - - 2 1
```

| tratto | mossa | notazione | esito partita | eval |
|---|---|---|---|---|
| bianco | b5e2 | Qe2 | bianco vince | — |

### 12. [val]

```
8/pR3p2/1b2b1p1/6k1/8/1r3BK1/2R3P1/8 w - - 12 1
```

| tratto | mossa | notazione | esito partita | eval |
|---|---|---|---|---|
| bianco | c2d2 | Rd2 | nero vince | — |

### 13. [val]

```
1r2rbk1/p4pp1/2p1b2p/4q3/N3P3/P3R1P1/1PQ3BP/4R2K w - - 4 1
```

| tratto | mossa | notazione | esito partita | eval |
|---|---|---|---|---|
| bianco | b2b4 | b4 | nero vince | — |

### 14. [val]

```
2r2rk1/p5pp/1p2p1n1/n2pP3/3P2P1/P1Pb2B1/3N2BP/R1R3K1 b - - 4 1
```

| tratto | mossa | notazione | esito partita | eval |
|---|---|---|---|---|
| nero | a5c4 | Nc4 | bianco vince | -152 |

### 15. [test] ← matto

```
4k3/p4p2/4b3/2b5/5B1P/5BP1/4Q1K1/1q6 b - - 2 1
```

| tratto | mossa | notazione | esito partita | eval |
|---|---|---|---|---|
| nero | b1g1 | **Qg1#** | nero vince | — |

### 16. [test]

```
8/3n4/4k3/1P1Q2p1/3K4/6P1/7P/8 b - - 4 1
```

| tratto | mossa | notazione | esito partita | eval |
|---|---|---|---|---|
| nero | e6e7 | Ke7 | bianco vince | — |

### 17. [test]

```
rnb1kb1r/pp1p1ppp/4p3/q2BP3/8/2N2N2/PB3PPP/R2QK2R b KQkq - 2 1
```

| tratto | mossa | notazione | esito partita | eval |
|---|---|---|---|---|
| nero | f8b4 | Bb4 | nero vince | 196 |

### 18. [test]

```
rn1q1rk1/pp2ppbp/2p2np1/8/2BP1Bb1/2N1PN1P/PP3PP1/R2QK2R b KQ - 0 1
```

| tratto | mossa | notazione | esito partita | eval |
|---|---|---|---|---|
| nero | g4f3 | Bxf3 | patta | — |

### 19. [test]

```
r4rk1/p1p4p/1p2p1q1/4Pp2/4b1p1/PQB1P3/1P3PPP/3R1RK1 w - - 2 1
```

| tratto | mossa | notazione | esito partita | eval |
|---|---|---|---|---|
| bianco | d1d7 | Rd7 | nero vince | -50 |

### 20. [test] ← matto forzato nell'eval

```
6Q1/K7/4p3/8/3P4/2P4p/6k1/8 b - - 2 1
```

| tratto | mossa | notazione | esito partita | eval |
|---|---|---|---|---|
| nero | g2f2 | Kf2 | bianco vince | 8115 |

---

## Esito dell'ispezione

_Da compilare dopo aver guardato i campioni._

- [ ] Le mosse sono sensate, non solo legali
- [ ] Il tratto nella FEN corrisponde al colore che muove
- [ ] Gli esiti sono misti
- [ ] Il campione 15 e davvero matto
- [ ] Il campione 3 e un'apertura riconoscibile

**Verificato da:** ______  **Data:** ______

## Rigenerare con altri campioni

```
python scripts/verify_dataset.py --data data/processed --inspect 20
```

Cambia `--seed` per un campione diverso.
