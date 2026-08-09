# Diario di lavoro

Note su cose non ovvie imparate strada facendo. Non e un log di commit — quello lo fa
git. Qui vanno i "non me lo aspettavo" e i bug che hanno richiesto tempo per essere
capiti, perche sono esattamente quelli che si ripresentano.

Regola #6: se scopri un bug, il primo commit del fix contiene il test che lo avrebbe
intercettato. Annota qui **quanto tempo e costato trovarlo** — e la misura di quanto
valeva quel test.

---

### 2026-08-07 — Setup del progetto
Struttura, pipeline di check e ambiente definiti prima di scrivere codice di dominio.
La scelta e deliberata: in questo progetto i bug non crashano (encoder sbagliato, segno
invertito nel backup MCTS, leakage nello split), quindi l'infrastruttura di verifica
deve esistere prima delle cose da verificare.

Prossimo passo: Stadio 1 — encoder e codifica mosse, con la batteria di test §2.5 come
Gate 1.

---

### 2026-08-08 — Stadio 1: l'encoder e passato al primo colpo, i test no

L'encoder e la codifica mosse hanno passato la batteria §2.5 senza bug veri. I due
fallimenti costati tempo erano **entrambi nei test**, non nel codice — e sono istruttivi
proprio per questo.

**1. La simmetria non e quella che sembra.** Il primo test scritto era
`evaluate(b) == -evaluate(b.mirror())`, e falliva su 15.910 posizioni su 16.149. Il
codice era corretto: `chess.Board.mirror()` specchia la posizione, scambia i colori **e
inverte il tratto**. E quindi la stessa situazione vista dall'altro lato, e in
convenzione negamax deve valere **uguale**, non opposta. L'antisimmetria vale per una
trasformazione diversa: stessa posizione, solo il tratto invertito.

Il piano parla di "simmetria specchiata" e di "simmetria colori" come se fossero un'unica
cosa. Sono due proprieta distinte, e servono entrambe:

| Trasformazione | Proprieta attesa |
|---|---|
| `board.mirror()` (posizione + colori + tratto) | `eval(b) == eval(mirror(b))` |
| solo il tratto invertito | `eval(b) == -eval(flip_turn(b))` |

Ora sono due test separati, con la spiegazione nel docstring. Un test sbagliato che
fallisce costa un'ora; un test sbagliato che *passa* costa il progetto.

**2. `Board.fen()` normalizza l'en passant.** Il round-trip FEN falliva su
`rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq e3 0 1`: l'encoder codificava la
colonna e, la ricostruzione la perdeva. Causa: `fen()` **cancella** il campo en passant
quando nessun pedone puo davvero catturare al varco, mentre `fen(en_passant="fen")` lo
conserva. Si stavano confrontando due normalizzazioni diverse. Il tensore era giusto.

**Prezzo del biglietto:** ~40 minuti in due, quasi tutti spesi a cercare un bug
nell'encoder che non c'era. La lezione che vale per il resto del progetto: quando un
test fallisce su una proprieta matematica, verificare *l'enunciato* prima
dell'implementazione.

---

### 2026-08-08 — Stadio 2: la baseline vinceva e non concludeva

Qui il bug era vero, ed e esattamente il tipo di bug che il piano prevede: nessuna
eccezione, nessun crash, solo un motore che gioca male.

**Sintomo:** contro l'avversario casuale, partite a +683 di valutazione (alfiere e
cavallo contro un pedone) che finivano **patta al limite delle 300 semimosse**. Con
KR vs K la baseline muoveva la torre avanti e indietro all'infinito.

**Due cause distinte, entrambe reali:**

1. **La ricerca non vedeva le ripetizioni.** `_negamax` controllava `is_repetition(3)`,
   ma a profondita 3 il matto e oltre l'orizzonte: tutte le mosse valevano uguale e il
   motore ne sceglieva una a caso, ripetendo. Corretto trattando come patta gia la
   **seconda** occorrenza (`is_repetition(2)`), che e la convenzione standard dei motori:
   se una linea porta a ripetere, l'avversario puo sempre insistere fino alla terza.

2. **Mancava l'incentivo al matto.** Le PST del re non bastano: servono i due termini
   classici — spingere il re debole verso il bordo, avvicinare il re forte. Aggiunti in
   `_mate_drive`, attivi solo con un vantaggio da finale vinto (>= una torre) e pesati
   sul peso di finale.

**Esito:** KRR vs K ora da matto; KR vs K e KQ vs K arrivano alla regola delle 50 mosse
senza concludere. **Non e stato corretto oltre**: e un limite noto e accettabile a
1200-1400 Elo, ed e registrato in RESULTS.md come tale invece di essere nascosto.

**Falsa pista da annotare:** durante la diagnosi ho letto piu volte `board.result()` su
partite ancora in corso, interpretando `1/2-1/2` come "finita patta" quando era il mio
loop diagnostico a esaurire le iterazioni. Tre volte, prima di accorgermene. Quando si
debugga con script usa e getta, **stampare sempre il motivo dell'uscita dal ciclo**, non
solo lo stato finale.

---

### 2026-08-08 — Il bug vero: finestra alpha-beta stretta alla radice

> Le due voci sopra descrivono sintomi **reali ma secondari**. La causa dominante era
> un'altra, e le ha in gran parte mascherate. Questa e la voce che conta.

**Come e emerso.** Il match ufficiale del Gate 2 ha dato **14W-9L-77D su 100** contro
l'avversario *casuale*: Elo +17 ± 68, statisticamente indistinguibile da zero. Un
minimax a profondita 3 con quiescence deve vincerne ~100 su 100. Non era un limite dei
finali: la baseline **perdeva partite contro mosse casuali**.

**Il bug.** Nel ciclo alla radice di `search()`:

```python
score = -_negamax(work, depth - 1, -INFINITY, -alpha, stats, ...)   # SBAGLIATO
```

Restringere la finestra con `alpha` e corretto quando serve sapere solo *quale* mossa e
la migliore — i rami peggiori vengono potati e la ricerca costa molto meno. Ma il
punteggio che quei rami restituiscono **non e il loro valore vero**: e il limite a cui
la potatura si e fermata.

E la radice quei punteggi li usa per un secondo scopo: raccogliere le mosse di pari
merito in `best_moves`, da cui `rng` estrae. Misurato su una posizione di test: mosse
che valevano **−942** tornavano **+125**, il valore di cutoff. Finivano nel gruppo delle
"migliori" ed erano estratte a caso.

**Il sintomo e cio che lo rendeva difficile.** La baseline non regalava mai un pezzo in
modo vistoso — nessun singolo crollo di valutazione da inseguire nelle partite. Sceglieva
a caso fra mosse che *credeva* equivalenti, e il vantaggio si dissolveva nell'arco di
venti mosse. Ho cercato il bug in `_mate_drive`, nella rilevazione delle ripetizioni,
nell'ordinamento MVV-LVA e nella quiescence prima di guardare la riga giusta.

**Il fix:** finestra piena `(-INFINITY, INFINITY)` ad ogni mossa della radice. Costa
circa 2-3x nei nodi visitati — accettabile a questa profondita, e la potatura vera resta
dentro `_negamax`. Effetto sul match ufficiale, stesso seed e stesso setup:

| | Partite | Risultato | Elo |
|---|---|---|---|
| Prima | 100 | 14W-9L-77D (52.5%) | +17 ± 68 |
| Dopo | 100 | **100W-0L-0D (100%)** | non stimabile |

Una riga di codice fra un motore indistinguibile dal caso e un cappotto.

**Nota su cosa NON era il problema.** Le due voci precedenti (finali non convertiti,
ripetizioni) descrivono correzioni reali e utili, ma marginali: con la finestra rotta la
baseline non arrivava quasi mai a un finale vinto. Averle scritte per prime ha allungato
la diagnosi — cercavo un difetto di *tecnica scacchistica* mentre il difetto era di
*algoritmo*. Quando i sintomi sono diffusi (nessun singolo errore vistoso, degrado
graduale ovunque), la causa e piu probabilmente strutturale che di dominio.

**Il test che lo avrebbe intercettato** (regola #6):
`test_le_mosse_pari_merito_sono_davvero_equivalenti` — su una posizione con una sola
mossa buona, rilancia la ricerca con 12 seed diversi e pretende sempre la stessa mossa.
Verificato che fallisce sul codice bacato: reintrodotto il bug apposta, il test e
diventato rosso, poi ripristinato.

**Le due lezioni.**

1. **Un valore potato non e un valore.** Vale ovunque si riusino i punteggi di
   alpha-beta per qualcosa di diverso dal scegliere il massimo — ordinamento, campionamento,
   analisi multi-PV. Al Gate 5 la stessa trappola si presentera con le statistiche dei
   nodi MCTS.
2. **Il match contro l'avversario casuale e un test diagnostico, non una formalita.**
   Nessun test unitario aveva colto il problema: le tattiche elementari passavano tutte,
   perche su una posizione con una cattura ovvia la prima mossa esaminata e gia la
   migliore e la finestra non fa danni. Solo giocare partite intere lo ha rivelato.

---

### 2026-08-09 — Stadio 3: il parsing era 90 volte piu lento del necessario

**Come e emerso.** Prima di lanciare la conversione ho misurato la velocita su un
campione del dump vero: 51 posizioni/s. Fatto il conto: **27 ore per 5 milioni di
posizioni**. Non un numero da scoprire il giorno dopo guardando un run ancora in corso.

**La causa.** `chess.pgn.read_game` costruisce l'albero completo delle mosse di ogni
partita. Ma i filtri di §2.1 scartano il **99% delle partite** (misurato: 1,02% tenute
sul 2026-07), e lo scarto avviene guardando i soli header. Si stava quindi parsando la
notazione di 26 milioni di partite per buttarne via 26,7 milioni.

**Il profiling** (regola #4 — niente ottimizzazione senza misura):

| | Partite/s |
|---|---|
| sola lettura righe | 22.842 |
| `read_headers` + salto manuale | 1.661 |
| visitor con `SKIP` | 1.702 |
| `read_game` completo | **94** |

**Il fix.** `_FilteringVisitor` applica i filtri in `end_headers` e restituisce
`chess.pgn.SKIP` quando la partita non passa: il parser salta il corpo senza costruirlo.
Da 51 a **4.590 posizioni/s** end-to-end, fattore 90. La conversione di 20M posizioni e
durata 114 minuti invece delle ~5 ore che sarebbero servite.

**La strada sbagliata, che sembrava la piu ovvia.** Il primo tentativo era
`read_headers` per filtrare, poi saltare a mano il corpo delle partite scartate. Non
funziona, e per un motivo non documentato in modo evidente: **`read_headers` consuma gia
il corpo** e si ferma sull'`[Event` successivo. Lo stream non e posizionato sulle mosse,
quindi non c'e nulla da saltare — e ogni tentativo di "saltare" mangia la partita dopo.
Sintomo: 3 partite lette invece di 5.

Ci ho provato due volte, con due implementazioni diverse dello stesso salto manuale,
prima di verificare cosa lasciasse davvero nello stream. **Verificare l'assunzione
costava una riga** (`read_headers` poi `readline`), e l'ho fatto solo dopo il secondo
fallimento.

I test su PGN sintetici hanno intercettato entrambi i tentativi rotti prima che
toccassero il dump vero. Sono cinque partite scritte a mano in un file: hanno appena
ripagato il tempo di scriverle.

**Nota sul dimensionamento.** Ho lanciato la conversione completa e l'ho fermata dopo
tre minuti: la proiezione dava 47-78M posizioni in 3,6-6 ore. Il piano dice
esplicitamente che oltre i 15-20M non serve — una rete da 3,5M parametri satura prima.
Rilanciata con `--target-positions 20000000`. **Fermare un run che sta funzionando** e
controintuitivo, ma tre minuti di proiezione hanno risparmiato quattro ore di calcolo
per dati che non sarebbero stati usati.
