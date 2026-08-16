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

---

### 2026-08-13 — Stadio 4: tre bug nella loss, e la value head che non impara

Il training e andato liscio: 12 epoche, 11,8 ore, top-1 al 52,11%. I problemi sono stati
tutti **prima** di lanciarlo, ed e esattamente dove dovevano stare.

**1. Il label smoothing spalmato sulle mosse illegali.**

`F.cross_entropy(..., label_smoothing=0.05)` distribuisce la massa di smoothing su TUTTE
le classi. Con 4672 mosse di cui ~30 legali, le altre 4642 sono a -1e9 per la
mascheratura, e il conto e:

    0.05 / 4672 * 4642 * 1e9 = 49.678.938

La loss valeva **49.683.864** invece di 3,40 — che e `ln(30)`, il valore corretto per 30
mosse equiprobabili.

Il gradiente sui logit legali restava giusto, quindi **l'overfit test passava lo stesso
al 100%**. Il bug non impediva alla rete di imparare: rendeva impossibile *accorgersi* se
stava imparando. Il termine WDL, che vale ~0,9, sarebbe stato schiacciato di sette ordini
di grandezza, e ogni soglia numerica del Gate 4 priva di senso.

E il tipo di bug peggiore: passa i test, non crasha, e falsifica solo gli strumenti con
cui lo cercheresti.

**2. La sentinella -1e9 non entra in fp16.**

Il massimo rappresentabile in mezza precisione e 65504. Con AMP attiva — il default sulla
3050 — i logit arrivano alla mascheratura in fp16 e PyTorch solleva
`RuntimeError: value cannot be converted to type c10::Half without overflow`. Crash al
primo batch.

I test unitari non lo vedevano perche girano tutti in fp32. **L'ha trovato lo smoke test**
da 60 step, che costa un minuto e gira sul percorso vero. Ora la sentinella e
`finfo(dtype).min / 2`, e i test sono parametrizzati sui tre dtype.

**3. Lo stato RNG caricato su CUDA.**

`torch.load(..., map_location="cuda")` sposta *tutti* i tensori sulla GPU, stato dei
generatori compreso. Ma `set_rng_state` pretende un ByteTensor su CPU: `TypeError` alla
ripresa, cioe **proprio quando il checkpoint serve**.

Anche questo trovato provando la ripresa sul serio, non leggendo il codice.

**La lezione comune ai tre.** I test unitari coprono la logica; lo smoke test copre
l'integrazione con l'hardware vero (dtype, device, memoria). Due dei tre bug erano
invisibili ai primi e ovvi al secondo. Sessanta step di training finto prima di undici
ore di training vero sono il rapporto costi-benefici migliore di tutta la sessione.

---

### 2026-08-13 — La value head impara quasi nulla: numeri

Il criterio del Gate 4 dice "loss del valore ~0.45" senza specificare quale delle due
misure. Misurate entrambe, il numero che conta e un terzo:

| | CE |
|---|---|
| Predire a caso | 1,0986 |
| Predire sempre la distribuzione media del dataset | 0,9710 |
| **La nostra rete** | **0,9038** |

**0,067 nats di guadagno** su una strategia che ignora completamente la posizione. La
rete e appena meglio di chi risponde sempre "45% vittoria, 12% patta, 43% sconfitta".

Non e un bug: e la criticita #5 che si manifesta come previsto. L'esito della partita e
un'etichetta rumorosa — la stessa posizione vinta compare come "vittoria" o "sconfitta"
a seconda di cosa succede venti mosse dopo, e la rete non ha modo di distinguere.

**Cosa aspettarsi allo Stadio 5.** L'MCTS usa la value head per valutare le foglie. Con
una value head cosi debole, la ricerca rischia di essere guidata da rumore. Il piano
prevede gia la diagnosi §4.4: se policy+MCTS a 400 simulazioni non guadagna almeno 150
Elo sulla policy pura, servono le mitigazioni.

Tre strade, in ordine di costo crescente, se la diagnosi va male:
1. **pesare di piu l'eval Stockfish** nella loss — c'e sul 34% dei campioni ed e molto
   meno rumoroso dell'esito
2. **filtrare le posizioni finali** dal target WDL: nelle ultime dieci mosse l'esito e
   quasi deterministico e la rete potrebbe impararlo li
3. accettare che l'MCTS sia guidato prevalentemente dalla policy, che invece funziona

---

### 2026-08-14 — Stadio 5: due bug nel batching, entrambi di segno opposto al previsto

L'MCTS in se e passato quasi subito: matti in 1 trovati 50/50 gia alla prima esecuzione,
simmetria colori a posto, backup corretto. I due bug veri stavano tutti e due nella
**ottimizzazione** — il batching delle foglie — e sono istruttivi perche entrambi
rendevano peggiore cio che doveva migliorare.

**1. Il batching rendeva la ricerca cieca.**

Una foglia messa in coda per la valutazione resta **non espansa**. Il ciclo di selezione
scende finche `node.is_expanded` e falso, quindi le simulazioni successive tornavano
sulla stessa foglia e la rimettevano in coda: il batch si riempiva di N copie della
stessa posizione invece di esplorare rami nuovi.

Misurato su un matto in 1 con 39 mosse legali e 50 simulazioni:

| `batch_size_leaves` | terminali trovati | mossa giocata |
|---|---|---|
| 1 | 43 | il matto |
| 8 | **0** | una a caso |

L'ottimizzazione che doveva rendere la ricerca sei volte piu veloce la stava rendendo
incapace di vedere un matto in una mossa.

**2. Il virtual loss era un premio, non una penalita.**

Il primo fix — tenere un insieme dei nodi gia in coda — ha risolto la cecita ma ha
ridotto il batch a una posizione: 199 collisioni su 200 simulazioni. Chiudevo il batch
alla prima collisione invece di provare un'altra strada.

Il secondo tentativo — aumentare il virtual loss e riprovare — e andato **peggio**:
24.433 collisioni. A quel punto ho smesso di correggere e ho guardato:

    nodo pulito      value =  0.0   puct = 0.1
    nodo penalizzato value = -1.0   puct = 1.1

Il virtual loss abbassava `child.value`, ma il PUCT usa `-child.value` — perche il valore
del figlio e dal punto di vista dell'avversario. **Sottraendo, la penalita cambiava
segno e diventava un bonus.** Piu penalizzavo un ramo, piu la selezione ci tornava.

Il fix e un carattere: `value_sum - virtual_loss` diventa `value_sum + virtual_loss`.

Effetto, 400 simulazioni con la rete vera:

| batch | tempo | sim/s | chiamate GPU |
|---|---|---|---|
| 1 | 1,88 s | 213 | 401 |
| 24 | 0,31 s | **1.276** | 21 |

**La lezione comune.** Tutti e due i bug erano nel *codice di ottimizzazione*, non
nell'algoritmo. L'MCTS "puro" — selezione, espansione, backup — ha funzionato al primo
colpo. Il batching, che esiste solo per andare piu veloce, ha richiesto tre tentativi e
ha prodotto un motore cieco e uno che si autosabotava.

Vale il corollario della regola #4 (niente ottimizzazione senza profiling): **ogni
ottimizzazione va verificata contro la versione lenta**, non solo cronometrata. Il test
`test_il_batching_non_impedisce_di_vedere_i_terminali` confronta batch 1, 4, 8 e 24 sulla
stessa posizione e pretende lo stesso risultato — se ci fosse stato dall'inizio, avrei
saltato entrambi i giri.

**Tre errori miei nei test, per completezza.** Prima di trovare i bug veri ho corretto
tre volte il test "MCTS batte la policy pura", ogni volta convinto di aver trovato il
problema:

1. rete finta con valore costante zero — l'albero non ha nulla da ottimizzare, ripete
2. `policy_player` con `max()` su priori uniformi — giocatore deterministico degenere,
   muoveva la torre h8-g8 all'infinito
3. `max_plies=160` — troncava partite in cui l'MCTS aveva **il doppio dei pezzi**

Il terzo e **lo stesso errore gia commesso al Gate 2** con la baseline. Non l'ho
riconosciuto subito: ho guardato il risultato ("6 patte su 6") invece dello stato finale
delle partite. Bastava stampare il materiale.

---

### 2026-08-14 — La value head debole non ha impedito nulla

Al Gate 4 la diagnosi era severa: CE 0,9038 contro 0,9710 di chi ignora la posizione,
0,067 nats di guadagno. La domanda aperta era se una value head cosi povera rendesse
l'MCTS inutile, dato che e lei a valutare le foglie.

**Risposta: no.** Diagnosi §4.4 su 200 partite: **+486 Elo ± 103**, tre volte la soglia
di 150 che il piano fissava come limite per intervenire.

La spiegazione, a posteriori ovvia: la ricerca guadagna forza **anche solo esplorando**.

- i terminali (matto, stallo, patta) sono risposte **esatte** e non passano dalla rete
- vedere una cattura tre semimosse avanti non richiede valutazione posizionale raffinata
- la policy, che invece funziona bene (top-1 51,66%), guida la selezione verso i rami
  giusti, e la ricerca deve solo confermare o smentire

Il valore serve a ordinare fra alternative che la tattica non risolve. Li e debole — e si
vedra nel gioco posizionale a lungo termine, non nelle 400 simulazioni di una mossa.

**Nessuna delle mitigazioni §4.4 e stata applicata**, perche non ce n'e stato bisogno.
Restano disponibili per lo Stadio 6, dove il target del valore migliora da solo:
nell'Expert Iteration il valore da imparare e il risultato della ricerca, non l'esito
grezzo della partita.

---

### 2026-08-16 — Stadio 6: il "+380 Elo" era falso di un fattore 3

Il ciclo RL e andato bene: 25 iterazioni, 7 promozioni, 20,8 ore, patte al 16%. Ma il
numero che il riepilogo stampava era **sbagliato**, e la storia di come me ne sono
accorto vale piu del risultato.

**Cosa diceva lo script.** `summary.json` sommava le stime Elo dei singoli gating:
+41, +83, +53, +41, +41, +58, +64 = **+380 Elo cumulativo**.

**Perche e falso.** Due errori indipendenti, entrambi bastanti da soli:

1. **Gli intervalli si accumulano.** Ogni gating aveva IC ±88 su 60 partite. Sommandone
   sette, l'incertezza sale a ~±233 — piu del doppio del guadagno reale. Un "+380 ± 233"
   non e un'affermazione, e un'ammissione di ignoranza.
2. **Misura la cosa sbagliata.** Ogni gating confronta la rete nuova con la
   **precedente**, non con quella di partenza. I guadagni relativi non compongono: se A
   batte B di 40 Elo e B batte C di 40, A **non** batte C di 80.

**Il numero vero.** Match diretto fra rete finale e rete supervisionata, 200 partite:
**101W-35L-64D, +119 ± 51 Elo**. L'intervallo sta tutto sopra lo zero, quindi il
miglioramento e dimostrato — ma e **un terzo** di quanto il riepilogo suggeriva.

**Il segnale che avrebbe dovuto insospettirmi prima.** Nessun gating aveva superato la
soglia SPRT: LLR massima +0,89 contro 2,94. Tutte e sette le promozioni erano avvenute
per la regola di fallback (≥55% a fine partite), non per evidenza statistica. Con 60
partite per gating l'SPRT non ha abbastanza campione per decidere, e il fallback e piu
permissivo: promuove al 55% anche quando l'IC va da -30 a +110.

Avevo quel dato sotto gli occhi ad ogni iterazione e non l'ho letto come un avvertimento.

**La lezione, che e la regola #1 in forma piu severa.** Non basta che ogni numero abbia
un intervallo di confidenza: bisogna anche chiedersi **se sommare quei numeri abbia
senso**. Sette misure rumorose della stessa quantita si mediano e l'incertezza cala;
sette misure rumorose di quantita *diverse* si sommano e l'incertezza cresce.

Il codice ora riporta entrambi i numeri, con `measure_rl_gain.py` che spiega nel
docstring perche il primo non va usato. Non ho tolto il cumulativo dal riepilogo: e utile
per vedere l'andamento, purche si sappia cos'e.

---

### 2026-08-16 — La value head si e sistemata da sola

Al Gate 4 la value head era il punto debole misurato del progetto: CE 0,9038 contro
0,9710 di chi ignora la posizione, appena 0,067 nats di guadagno. Al Gate 5 avevamo
verificato che non impediva all'MCTS di funzionare (+486 Elo sulla policy pura), ma
restava debole.

**L'Expert Iteration l'ha dimezzata**: value loss da 0,3406 a 0,1481 in 25 iterazioni.

Il motivo e strutturale, non fortuna. Nel training supervisionato il target del valore
era l'**esito della partita**: una posizione vinta etichettata "sconfitta" perche venti
mosse dopo qualcuno ha sbagliato. Rumore puro. Nell'Expert Iteration il target diventa
`0.4 * Q_radice + 0.6 * z` (§5.5): Q e la stima prodotta dalla ricerca appena fatta su
quella posizione, molto meno rumorosa dell'esito finale.

E la conferma sul campo di cio che §5.5 prometteva — "costa una riga di codice", ed e
vero: la riga e `value_target()` in `search/parallel.py`.

**Cosa resta aperto.** La value head e migliorata ma non e forte in assoluto: non l'ho
rimisurata contro le baseline banali dopo l'RL, e sarebbe la verifica onesta da fare
prima di dire che il problema e risolto.
