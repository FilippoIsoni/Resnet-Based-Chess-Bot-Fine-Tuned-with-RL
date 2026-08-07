# Checklist operativa

Versione condensata di [PIPELINE.md](../PIPELINE.md), da tenere aperta mentre si lavora.

## Ogni task

```
[ ] Test scritto PRIMA (per encoder e codifica mosse non e dogma: e l'unico modo di sapere se funziona)
[ ] Implementazione minima che lo fa passare
[ ] check.py --level 1 verde
[ ] Verifica INCROCIATA con una fonte indipendente dall'implementazione
[ ] Commit: il messaggio dice cosa e stato VERIFICATO, non solo cosa e stato scritto
[ ] JOURNAL.md aggiornato se ho imparato qualcosa di non ovvio
```

## Ogni sessione di training

```
[ ] python scripts/preflight.py verde
[ ] Working tree pulito (altrimenti il commit hash nel checkpoint non basta a riprodurre)
[ ] Config salvata accanto ai pesi
[ ] Portatile alimentato e sollevato, power limit abbassato se e una sessione notturna
[ ] Checkpoint ogni N step attivo e testato (salva -> ricarica -> stessa loss)
```

## Ogni misura riportata

```
[ ] Intervallo di confidenza calcolato (con 200 partite: circa +-35 Elo)
[ ] >= 200 partite, aperture bilanciate da libro
[ ] Scritta in RESULTS.md con commit, config e checkpoint
[ ] Se negativa: scritta comunque
```

## Le quattro domande prima di dichiarare finito uno stadio

1. Il gate e verde, o l'ho dichiarato verde a mano?
2. Ho verificato il risultato contro una fonte **indipendente** dalla mia implementazione?
3. Se questo componente fosse rotto in modo silenzioso, quale test lo direbbe?
4. Fra due mesi saprei riprodurre questo numero?

## I bug che non crashano — controllo mirato

| Criticita | Sintomo | Test che lo intercetta |
|---|---|---|
| #1 flip scacchiera senza flip mossa | top-1 accuracy ferma al 25% invece di 50% | simmetria specchiata |
| #2 segno del valore nel backup MCTS | "sembra un po' debole" | simmetria colori |
| #3 split per posizione | validation loss ottimisticamente falsa | 0 game-id condivisi fra split |
| #4 promozione a donna nelle sottopromozioni | accuracy leggermente bassa | round-trip su mosse legali |
| #8 batching solo fra foglie | GPU al 5%, Fase 5 lentissima | profiling, 5x atteso |

Nessuno di questi produce un'eccezione. Sono tutti visibili **solo** come una metrica
che non sale.
