# Motore scacchistico con rete neurale — Piano definitivo

**Obiettivo:** motore di livello medio-alto, costruito hands-on applicando supervised learning e reinforcement learning. La forza massima non è il fine.
**Hardware:** RTX 3050 Laptop, 6 GB VRAM.
**Stack:** Python + PyTorch.

## Le cinque fasi

| # | Fase | Elo atteso | Tempo |
|---|------|-----------|-------|
| 1 | Baseline | ~1300 | 1-2 giorni |
| 2 | Preprocessing | — | 1 settimana |
| 3 | **Training supervisionato** | 1700-1900 | 1-2 settimane |
| 4 | MCTS | 2000-2300 | 1 settimana |
| 5 | **Reinforcement Learning** | +100-250 | 3-4 settimane |

Le fasi di apprendimento sono la **3** e la **5**. La 4 non allena nulla: costruisce la ricerca, che serve per giocare e — soprattutto — come generatore di etichette dentro la 5.

**Bot funzionante a fine Fase 4.** Le stime di Elo sono intervalli, non garanzie.

> **Distillazione da Stockfish: esclusa.** Darebbe +300-500 Elo per una notte di CPU, ma trasferisce per copia la conoscenza di un altro motore invece di produrne di nuova. Conseguenza da mettere in conto: la value head resta rumorosa, e siccome è lei a valutare le foglie dell'MCTS, la Fase 4 potrebbe rendere meno del previsto. Vedi §4.4.

---

# Fase 1 — Baseline

Minimax a profondità 3-4, valutazione materiale (P=1, N=B=3, R=5, Q=9) più piece-square tables. Circa 150 righe, gioca a **1200-1400 Elo**.

Non è un esercizio di riscaldamento: è il test di sanità dell'intero progetto. Se la rete neurale non batte questa baseline, c'è un bug, non un limite del modello.

### Infrastruttura da predisporre ora

- File di configurazione unico (YAML) con percorsi e iperparametri
- **Checkpoint ogni N step**: pesi + stato optimizer + posizione nel dataset + seed
- Logging su TensorBoard o Weights & Biases

Il checkpointing non è opzionale: su un portatile un aggiornamento di sistema o uno shutdown termico azzerano una notte di training.

### Metrica di riferimento

Stockfish con `UCI_LimitStrength=true` e `UCI_Elo` crescente: **1400 → 1800 → 2200**. Metro calibrato e gratuito.

---

# Fase 2 — Preprocessing

> **La fase più fragile del progetto.** Un bug qui non fa crashare niente: alleni su etichette sbagliate e te ne accorgi dopo giorni, guardando un'accuracy che non sale.

## 2.1 — Acquisizione dati

**Fonte principale:** database.lichess.org, dump mensili in PGN. In alternativa Lumbras Gigabase o Caissabase per partite di maestri.

Un file PGN è **testo**, non immagini: contiene la trascrizione delle mosse in notazione. La conversione in numeri è interamente compito tuo.

### Filtri

| Criterio | Valore | Motivo |
|----------|--------|--------|
| Elo medio | ≥ 2000 | Imitare gente forte, non media |
| Time control | rapid / classical | Il bullet è rumoroso |
| Esito | escludere tempo scaduto in posizione non persa | Etichetta di valore falsata |
| Annotazioni | preferire partite con `[%eval ...]` | Target di valore più pulito |

Circa il 6% dei PGN Lichess contiene già valutazioni Stockfish. Sono tecnicamente etichette di un altro motore, ma arrivano gratis nei dati che scarichi comunque. **Raccomandazione: tenerle** — avendo escluso la distillazione, sono l'unico correttivo disponibile per la value head.

### Aperture: si tengono, con capping

Il problema non è che non contino, ma che sono sovrarappresentate: la posizione iniziale compare in tutte le partite, e poche centinaia di posizioni si mangerebbero una fetta sproporzionata dei gradienti.

Soluzione: massimo 2000-5000 occorrenze per FEN distinta, oppure pesare ogni campione con `1/sqrt(conteggio_FEN)`.

Al momento di giocare si monta comunque un **libro Polyglot** (vedi §4.5), quindi la teoria d'apertura è garantita a prescindere.

### ⚠ Split del dataset

> **Splittare per PARTITA, mai per posizione.**
>
> Posizioni consecutive della stessa partita sono quasi identiche. Uno split casuale per posizione mette la stessa partita in train e validation, produce leakage, e la validation loss risulta ottimisticamente falsa: credi di generalizzare e stai memorizzando.

Split 95/2.5/2.5 a livello di partita, poi deduplicare per FEN tra gli split.

### Volume

**15-20 milioni di posizioni** (~200-250k partite) nella versione piena, **5 milioni** nella minimale. Oltre non serve: una rete da 3,5M parametri satura prima.

## 2.2 — Da PGN a posizioni

`python-chess` legge il PGN e rigioca le mosse una per una. Ad ogni passo hai in memoria un `board` corretto: da lì fai la fotografia logica dello stato.

Una partita di 40 mosse per parte produce ~80 posizioni. È così che 250.000 partite diventano 20 milioni di esempi.

**Perché le posizioni sono campioni indipendenti:** gli scacchi sono markoviani a informazione perfetta. La posizione attuale contiene tutto il necessario per decidere la mossa — come ci sei arrivato è irrilevante, tranne per arrocco, en passant e ripetizioni, che si codificano come piani espliciti. Non serve alcun modello sequenziale.

Anzi, **mescolare le posizioni fra partite diverse è necessario**: due posizioni consecutive sono quasi identiche, quindi un batch di mosse consecutive avrebbe gradienti fortemente correlati e il training sarebbe instabile. In un batch da 512 vuoi 512 partite diverse.

L'unico punto in cui la sequenza entra è il **target del valore**: l'etichetta "questa posizione ha portato a una vittoria" viene dal futuro della partita. Ma entra come etichetta, non come input.

## 2.3 — Encoder posizione → tensore 19×8×8

| Piani | Contenuto |
|-------|-----------|
| 0-5 | Pedoni, cavalli, alfieri, torri, donne, re — **lato che muove** |
| 6-11 | Stessi sei tipi — **lato avversario** |
| 12-15 | Diritti di arrocco (corto/lungo × mio/avversario) |
| 16 | Colonna dell'en passant |
| 17 | Contatore delle 50 mosse, normalizzato |
| 18 | Conteggio delle ripetizioni della posizione |

**Nessun piano per il turno:** se la scacchiera è sempre orientata dal lato di chi muove, quel piano varrebbe costantemente 1 e non trasporterebbe informazione.

Non chiamarlo embedding: un embedding è una rappresentazione *appresa*, questa è una codifica deterministica decisa a mano, più simile a un one-hot.

### Orientamento

**Orientare sempre dal punto di vista di chi muove.** Se tocca al Nero: specchiare verticalmente e scambiare i colori, così la rete vede sempre "il mio esercito che sale". Dimezza lo spazio da imparare, a costo zero.

> ### ⚠ Bug silenzioso n.1 — il più comune di tutti
> **Specchiando la scacchiera bisogna specchiare ANCHE l'indice della mossa-etichetta.**
>
> Fare l'uno e dimenticare l'altro significa allenare su etichette sistematicamente sbagliate per metà del dataset. Nessun errore a runtime: l'unico sintomo è la top-1 accuracy che si ferma intorno al 25% invece di salire verso il 50%.

## 2.4 — Codifica delle mosse → indice 0-4671

Schema AlphaZero: 8×8×73 = 4672.

- 56 "mosse da regina" per casella (8 direzioni × 7 distanze)
- 8 mosse di cavallo
- 9 sottopromozioni (3 pezzi × 3 direzioni)

**Attenzione:** la promozione a donna si codifica come normale mossa da regina, *non* nei piani di sottopromozione. Confonderle è la seconda fonte di bug più comune.

`python-chess` ti dà l'oggetto `Move`; la mappatura a indice è codice tuo.

## 2.5 — Batteria di test obbligatoria

| Test | Cosa verifica |
|------|---------------|
| 20 posizioni note con tensore atteso in JSON | Encoder corretto |
| Round-trip `move → indice → move` su 10.000 mosse legali | Codifica biiettiva |
| Simmetria specchiata (posizione bianca vs stessa specchiata nera) | Flip coerente fra input ed etichetta |
| Somma dei piani 0-11 = numero di pezzi | Nessun pezzo perso o duplicato |
| Confronto encoder vs re-parsing di 1000 posizioni | Nessuna deriva |

Far girare come test automatico ad ogni modifica dell'encoder.

## 2.6 — Storage

| Campo | Byte |
|-------|------|
| 12 bitboard `uint64` | 96 |
| Arrocco + en passant + ripetizioni | 2 |
| Indice mossa | 2 |
| Risultato | 1 |
| Eval Stockfish (se presente) | 2 |
| **Totale** | **~103** → **~2 GB per 20M posizioni** |

**Non salvare i tensori float espansi:** 20M × 19 × 64 × 4 byte = 97 GB. Si salva la forma compatta e si espande al volo nel DataLoader.

### Variante minimale (accettabile qui)

Il preprocessing non è saltabile — la conversione PGN → tensore *è* la generazione dell'input — ma l'ingegneria attorno sì.

| Ottimizzata | Minimale |
|---|---|
| Packing in bitboard su disco | Parsing PGN al volo nel DataLoader |
| Capping per FEN, filtri elaborati | Solo filtro Elo ≥ 2000 |
| Shard memory-mapped | Qualche file `.npy` |
| 15-20M posizioni | 5M posizioni |

Costo: training CPU-bound, da ~2 h/epoca a 6-8 h, GPU al 30%. Su 5M posizioni resta una notte. **L'ottimizzazione dello storage è data engineering, non machine learning**: da fare solo se diventa il collo di bottiglia reale.

### Prestazioni

python-chess parsa 500-2000 partite/s per core. `multiprocessing.Pool` con 6-8 worker → **4-8 ore** per la conversione completa. Si lancia una notte.

---

# Fase 3 — Training supervisionato

> **Prima fase di apprendimento.** Formalmente è *imitation learning*: nessun reward, nessuna interazione con l'ambiente, solo etichette esterne.

## 3.1 — Architettura

```
Input 19×8×8
  └─ Conv 3×3, 128 filtri, BN, ReLU
  └─ 8 × [blocco residuo: Conv3×3-BN-ReLU-Conv3×3-BN-(+skip)-ReLU]
       ├─ Policy head:  Conv 1×1 (32ch) → flatten → Linear → 4672 logit
       └─ Value head:   Conv 1×1 (8ch)  → Linear(128) → Linear(3) → softmax WDL
```

**~3,5M parametri.** In fp16 con batch 512 occupa ~1,5 GB di VRAM: la 3050 sta larga, si potrebbe salire a 10 blocchi × 160 filtri.

Le convoluzioni 3×3 hanno senso perché le relazioni scacchistiche sono locali e spaziali (catene di pedoni, case deboli, difese reciproche) e dopo 8 blocchi il campo recettivo copre l'intera scacchiera.

**Testa WDL (3 uscite softmax)** invece di un `tanh` scalare: la cross-entropy su tre classi dà gradienti più informativi della MSE, e distingue una posizione realmente pari da una sbilanciata con esito incerto — che con lo scalare valgono entrambe 0. Utile soprattutto qui, dove la value head è la componente debole.

**Entrambe le teste servono da subito.** Senza value head l'MCTS non può esistere: è lei che valuta le foglie. Allenarla ora evita di rifare tutto un mese dopo.

> ### ⚠ Bug silenzioso n.2 — convenzione di segno
> Il valore è espresso **dal punto di vista di chi muove**, quindi nel backup dell'MCTS va **negato ad ogni livello** risalendo l'albero.
>
> Sbagliarlo produce un motore che gioca *contro* se stesso, e il sintomo è solo "sembra un po' debole". Documentare la convenzione in un commento e testarla: la valutazione di una posizione e della stessa a colori invertiti deve essere simmetrica.

### Alternative scartate

- **NNUE:** più forte per watt, ma valuta soltanto (niente policy) e richiede un alpha-beta serio con quiescence, transposition table, move ordering. Molta più ingegneria, molto meno ML.
- **Transformer su 64 token:** scala benissimo (DeepMind, ~2900 Elo senza ricerca) ma con 270M parametri e miliardi di posizioni annotate. Alla nostra scala la ResNet vince per bias induttivo.
- **Pesi pretrained (Maia, reti lc0):** formato protobuf incompatibile con PyTorch, architettura vincolata, e Maia è costruita per *imitare* gli umani, blunder compresi. La Fase 3 costa una-due notti: non conviene.

## 3.2 — Loss

```
L = CE(policy_logits, mossa_giocata)
  + 0.5 · CE(wdl, risultato_partita)
  + 0.5 · MSE(value, eval_stockfish)     # solo dove disponibile
```

**Mascherare i logit delle mosse illegali a −inf prima della softmax**, in modo identico fra training e inferenza.

## 3.3 — Iperparametri

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

## 3.4 — Tempi e metriche

~2000-3500 posizioni/s → **1,5-2 ore per epoca** su 15M posizioni → ciclo completo in **una-due notti**. Se la GPU sta al 40%, il collo di bottiglia è il DataLoader.

| Metrica | Target | Nota |
|---------|--------|------|
| Top-1 accuracy policy (validation) | 45-52% | Non salirà oltre: gli umani non sono deterministici |
| Accuracy su 1000 tattiche Lichess | crescente | Il termometro più onesto |
| Loss del valore | ~0.45 | **Non è un fallimento**: è il rumore intrinseco dell'etichetta |

Sulla loss del valore: la stessa posizione può ricevere +1, 0 o −1 in partite diverse. Una posizione vinta al ventesimo tratto riceve etichetta −1 se al quarantesimo lasci la donna in presa.

## 3.5 — Risultato atteso

Rete che gioca a istinto (logit massimo, zero ricerca): **1700-1900 Elo**.

Non 2000, nonostante i dati siano di giocatori 2000: imitare la mossa *media* di una popolazione non equivale a giocare come quella popolazione. Quando due forti divergono, la rete impara una miscela sfocata, e nelle posizioni tattiche acute la sfocatura è fatale.

**Nota hardware:** limitare la potenza GPU (`nvidia-smi -pl`), batch moderato, portatile sollevato. Si perde il 10-15% di velocità e si evita il thermal throttling a metà notte.

---

# Fase 4 — MCTS

> **Non allena nulla: nessun peso viene aggiornato.** L'MCTS è *planning*, non RL — computa una mossa e butta via l'albero. Ma è il salto di Elo più grande del progetto, e non costa un solo step di training.

## 4.1 — Perché non è saltabile

Oltre a servire per giocare, l'MCTS è **il motore interno della Fase 5**. Nell'Expert Iteration le etichette su cui si allena la rete *sono* le distribuzioni di visite prodotte dalla ricerca. Senza MCTS non esistono target per la policy head, e l'unica alternativa è REINFORCE — che negli scacchi non converge: il reward è terminale e ternario dopo ~70 mosse, e il gradiente penalizza allo stesso modo le 68 mosse buone e i 2 blunder.

Se il costo pesa: **50-80 simulazioni** invece di 200. Il segnale peggiora ma resta incomparabilmente più informativo.

## 4.2 — PUCT

Ad ogni simulazione si scende nell'albero massimizzando:

```
Q(s,a) + c_puct · P(s,a) · sqrt(Σ_b N(s,b)) / (1 + N(s,a))
```

`P` dalla policy della rete, `Q` media dei valori osservati, `N` visite. Alla foglia si valuta **con la rete** e si propaga indietro negando il segno ad ogni livello. Si gioca la mossa più visitata. `c_puct` ≈ 1.5-2.5.

### Niente rollout casuali

Il Monte Carlo classico (giocare a caso fino alla fine) **fallisce negli scacchi**. Il Go è accumulativo: da una posizione vinta, giocare a caso finisce spesso in vittoria. Gli scacchi sono a morte improvvisa: con la donna in più, tre mosse casuali possono regalare il matto. Il rumore tattico distrugge il segnale posizionale.

La value head **è** ciò che rende praticabile la ricerca. Il "Monte Carlo" nel nome è un residuo storico: quello che resta è la parte *tree search*.

## 4.3 — Ottimizzazioni indispensabili

- **Batching delle foglie con virtual loss** — 16-32 posizioni per forward pass. Senza, la GPU resta ferma il 95% del tempo: 30 simulazioni/s invece di 400.
- Riconoscimento diretto di matto, stallo, ripetizione e regola delle 50 mosse **prima** di chiamare la rete.

**Collo di bottiglia atteso:** python-chess genera 20-50k mosse/s in puro Python. Sarà questo, non la rete, a limitare la profondità. Se il profiler lo conferma, riscrivere la generazione mosse in Cython o Rust — *solo dopo aver misurato*.

## 4.4 — Diagnosi precoce del rischio value head

Il rischio strutturale del piano (niente distillazione) si manifesta qui. **Misurarlo subito**, non a fine progetto:

1. Match a 200 partite fra *policy pura* e *policy + MCTS 400 simulazioni*
2. Se il guadagno è < 150 Elo, la value head non regge la ricerca

Mitigazioni in ordine:
- Aumentare il peso della loss del valore in Fase 3 (da 0.5 a 1.0) e riallenare le ultime epoche
- Sfruttare tutte le partite con annotazioni `[%eval]`
- Ridurre `c_puct` per fidarsi di più della policy e meno del valore
- Ultima risorsa: recuperare la distillazione

## 4.5 — Test di correttezza

Un MCTS bacato non crasha, sembra solo debole.

| Test | Criterio |
|------|----------|
| Matti forzati in 1 | Trovati sempre, anche a 50 simulazioni |
| Matti forzati in 2-3 | Trovati a 400-800 simulazioni |
| Rete casuale + MCTS vs rete casuale sola | La prima vince nettamente |
| Simmetria colori | Stessa valutazione a colori invertiti |
| Simulazioni crescenti | Forza monotonicamente crescente |

L'ultimo è il più diagnostico: se raddoppiare le simulazioni non aumenta la forza, c'è un bug oppure la value head è inutilizzabile.

## 4.6 — Libro d'aperture

Libro **Polyglot** (`.bin`) per le prime 8-12 mosse via `chess.polyglot`, ~20 righe. Costo computazionale zero, apertura di livello professionale garantita indipendentemente da cosa ha imparato la rete. È ciò che fanno tutti i motori seri, Stockfish compreso.

## 4.7 — Risultato atteso

**2000-2300 Elo.** Il bot è funzionante.

---

# Fase 5 — Reinforcement Learning

> **Seconda e ultima fase di apprendimento. Il cuore del progetto.**

## 5.1 — Metodo: Expert Iteration

**Non PPO, non REINFORCE.**

Il ciclo: il modello gioca contro se stesso con MCTS; per ogni posizione si salva la distribuzione delle visite π e il risultato z; si allena la rete su quei dati; si ripete.

**La loss è identica a quella della Fase 3.** Cambia solo la provenienza delle etichette: prima gli umani di Lichess, ora l'MCTS di se stessi. Non serve un algoritmo nuovo — si riusa il training loop esistente. Tutta la complessità sta nell'infrastruttura di generazione partite.

*Partire da pesi supervisionati invece che casuali non rende l'RL annacquato: l'algoritmo è lo stesso. Il "senza conoscenza umana" di AlphaZero era una tesi scientifica da dimostrare, non un requisito di efficienza. Senza pretraining, il cold start è insuperabile: due giocatori casuali non si danno mai matto, le partite finiscono per stallo o ripetizione, e il reward è quasi sempre patta — informazione zero.*

### Perché non PPO

- **Reward terminale e ternario** dopo 60-80 mosse: credit assignment quasi impossibile. L'MCTS dà invece un segnale **denso e locale** su ogni posizione.
- **Spazio azioni enorme e variabile** (4672 logit, ~35 legali, mascherati diversamente ogni volta): l'importance sampling è numericamente scomodo.
- **L'MCTS è un operatore di miglioramento della policy garantito**: π è dimostrabilmente meglio di p, perché è p *più* 200 simulazioni di verifica. Non serve stimare la direzione di miglioramento — la ricerca la fornisce. Vantaggio disponibile solo con un simulatore perfetto.

## 5.2 — Configurazione

| Parametro | Valore |
|-----------|--------|
| Partite per iterazione | 400-600 |
| Simulazioni per mossa | 150-200 (50-80 se il tempo stringe) |
| Rumore di Dirichlet (radice) | α = 0.3, peso 0.25 |
| Temperatura | 1.0 per le prime 15 mosse, poi ~0 |
| Replay buffer | ultime 8-10 iterazioni |
| Epoche per iterazione | 2-3 |
| Optimizer | SGD momentum 0.9 (più stabile di Adam sotto distribution shift) |
| Learning rate | **1e-4 o meno** |
| Penalità KL vs policy supervisionata | peso 0.05-0.1 |
| Gating | ≥55% su 200 partite, o test SPRT |

**La penalità KL** è un'aggiunta rispetto ad AlphaZero originale: loro partivano da zero e non avevano nulla da dimenticare; qui si parte da una rete che sa già le aperture umane e non deve buttarle via inseguendo strategie degeneri di self-play.

**Il gating** è l'unica cosa che impedisce di accorgersi dopo cinque giorni di essere peggiorati monotonicamente. Con 100 partite il rumore su un 50/50 è ±5%: la soglia al 55% sarebbe appena sopra il rumore. **Usare 200 partite**, o un SPRT che si ferma appena ha evidenza sufficiente.

## 5.3 — ⚠ Problema delle patte nel self-play

Man mano che la rete migliora, le partite fra due copie identiche diventano in maggioranza patte e il segnale evapora.

Soluzioni, da applicare tutte:
- **Randomizzare le aperture di partenza**: iniziare ogni partita da una posizione pescata a caso da un libro bilanciato (dopo 4-8 mosse). È ciò che fa Leela.
- Temperatura a 1.0 per le prime 15 mosse (già previsto)
- Monitorare la percentuale di patte: se supera il 70%, aumentare la diversità delle aperture

## 5.4 — ⚠ Batching fra partite, non solo fra foglie

La Fase 5 è il 90% del costo del progetto, e quasi tutto è generazione partite.

**Giocare 64-128 partite in parallelo** nello stesso processo, batchando le valutazioni di rete **fra partite diverse**, non solo fra foglie dello stesso albero. Con una partita per volta il batch è limitato dalla larghezza dell'albero; con 64 in parallelo si saturano batch da 256-512 e la GPU lavora davvero.

Guadagno stimato: **5-10×**. Da implementare *prima* di lanciare la prima iterazione.

## 5.5 — Target del valore migliorato

Passare dal risultato puro `z` a un **target misto**, come fa Leela:

```
target = alpha · Q_radice(s) + (1 - alpha) · z        con alpha in [0.3, 0.5]
```

`Q_radice` è già disponibile: è la media dei valori accumulati dalle simulazioni appena fatte per scegliere la mossa. **Costa una riga di codice.** È una stima *cercata*, molto meno rumorosa del risultato finale — concettualmente lo stesso trucco di TDLeaf(λ), e particolarmente utile qui dove la value head parte rumorosa.

## 5.6 — ⚠ Criterio di stop, da fissare PRIMA di iniziare

Un ciclo RL senza condizione di abbandono definita in anticipo diventa un pozzo di tempo.

- **Gating mai passato dopo 10 iterazioni** → fermarsi e analizzare, non insistere
- **Guadagno cumulativo < 50 Elo dopo 25 iterazioni** → chiudere la fase
- **Tetto duro: 40 iterazioni.** Oltre, si sta ottimizzando invece di imparare

Guadagno realistico: **+100-250 Elo per 3-4 settimane di macchina**. Si fa per valore didattico e di portfolio — un ciclo di Expert Iteration implementato è un ottimo pezzo da mostrare — non per efficienza.

## 5.7 — Vincolo pratico

3-4 settimane di macchina accesa su un portatile che serve anche per studiare. Opzioni:
- Generazione solo di notte, checkpoint a ogni iterazione
- Spostare la generazione su **Kaggle** (sessioni 9 h, esecuzione in background), scaricando i buffer e allenando in locale
- Ridurre la scala: 200 partite/iterazione a 80 simulazioni, accettando più iterazioni

## 5.8 — Variante economica: TDLeaf(λ)

Invece di un'iterazione di self-play completo: 2000 partite veloci contro Stockfish limitato, aggiornando **solo la value head** con TDLeaf(λ=0.9) — bootstrap sulla valutazione della foglia della variante principale, non della radice.

Costa una frazione perché non serve un MCTS profondo, e dice se il collo di bottiglia è la valutazione o la policy. Il grafico della loss prima/dopo è materiale di analisi presentabile.

*Precedente storico: TD-Gammon (1992) raggiunse livello mondiale a backgammon con TD(λ); KnightCap (1999) e Giraffe (2015, livello Maestro Internazionale) usarono TDLeaf(λ) negli scacchi.*

---

# Riepilogo delle criticità

| # | Criticità | Contromisura | Fase |
|---|-----------|--------------|------|
| 1 | Flip scacchiera senza flip mossa | Test di simmetria specchiata | 2 |
| 2 | Segno del valore nel backup MCTS | Test di simmetria colori | 3, 4 |
| 3 | Split train/val per posizione → leakage | Split **per partita** + dedup FEN | 2 |
| 4 | Promozione a donna confusa con sottopromozione | Test round-trip su mosse legali | 2 |
| 5 | Value head debole senza distillazione | Diagnosi §4.4 + mitigazioni graduate | 4 |
| 6 | python-chess collo di bottiglia nell'MCTS | Profilare prima, riscrivere solo se confermato | 4 |
| 7 | Patte nel self-play | Aperture randomizzate da libro | 5 |
| 8 | Batching solo fra foglie | 64-128 partite in parallelo | 5 |
| 9 | RL senza criterio di stop | Soglie a 10 / 25 / 40 iterazioni | 5 |
| 10 | Tempo macchina su portatile condiviso | Generazione notturna o su Kaggle | 5 |

---

# Ordine di attacco

1. **Encoder + codificatore mosse + batteria di test** (§2.3-2.5) — tutto il resto ci si appoggia
2. **Baseline minimax** (Fase 1) — metro di paragone
3. **Pipeline dati completa** (Fase 2) — una notte di macchina
4. **Training supervisionato** (Fase 3)
5. **MCTS + libro + diagnosi §4.4** (Fase 4) → bot funzionante
6. **Expert Iteration** (Fase 5) — solo quando tutto il resto è stabile e misurabile

---

# Appendice A — Valutazione

Serve dalla Fase 3 in poi, e il gating della Fase 5 non funziona senza.

- **Stockfish** con `UCI_LimitStrength`, Elo crescente da 1320
- **200+ partite per livello**, con aperture bilanciate da libro per ridurre la varianza
- Elo relativo: `400 · log10(vittorie/sconfitte)`, o BayesElo
- Riportare sempre l'**intervallo di confidenza**: con 200 partite è circa ±35 Elo. Differenze inferiori non sono differenze.
- **Maia** (CSSLab) come sparring calibrato: modelli per fasce da 1100 a 1900 che sbagliano *in modo umano*, più informativi di Stockfish limitato che sbaglia in modo artificiale. Girano in lc0 con `go nodes 1`.
- Suite tattiche: STS, Arasan, WAC
- **lichess-bot** per un rating pubblico reale

# Appendice B — Deployment

**Flutter solo UI** (web app statica su GitHub Pages o Vercel) + **backend Python separato**. L'inferenza in-browser via WASM è scartata: 300 MFLOPs/posizione in WASM single-thread danno 7-10 posizioni/s, cioè 30 secondi per mossa. Su CPU nativa la stessa rete fa 150-350 posizioni/s → 200 simulazioni in 1-2 secondi.

**API:** `POST /move {fen, level, session}` → `{move, eval, pv, ms}`, più `GET /health`.
- `level` mappa sulle simulazioni (50/200/800): difficoltà regolabile, indispensabile perché il gioco resti divertente
- `session` abilita il riuso dell'albero MCTS (cache LRU): −30-40% di simulazioni

**Esportare in ONNX**, non spedire PyTorch: la wheel PyTorch CPU supera i 200 MB, `onnxruntime` sta in ~15. Riusare la batteria di test della §2.5 per verificare che l'output combaci entro 1e-3.

**Hosting:** Fly.io (risveglio in 2-5 s) è la scelta consigliata. Oracle Cloud always-free è meglio ma richiede configurazione. Vercel serverless è da evitare per il backend: carico CPU-bound sostenuto, timeout stretti, si perde il riuso dell'albero.

**Tre trappole:** HTTPS obbligatorio (mixed content bloccato senza errori utili), CORS da autorizzare esplicitamente, cold start da mascherare chiamando `/health` all'apertura.

**Robustezza:** cap sulle simulazioni lato server, rate limit per IP, timeout a 10 s, validazione del FEN con python-chess.

**Flutter:** pacchetti `bishop` (regole) e `squares` (scacchiera); URL backend via `--dart-define`; interfaccia astratta `Engine` con `Future<String> bestMove(String fen)`.
