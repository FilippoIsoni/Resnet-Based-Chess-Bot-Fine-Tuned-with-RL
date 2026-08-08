# Decisioni

Registro delle scelte non ovvie e delle deviazioni dal piano. Serve a rispondere fra
due mesi alla domanda "perche l'avevo fatto cosi?".

Va aggiornato in due casi:
1. Si sceglie fra alternative non equivalenti.
2. **Si cambia o si allenta un criterio di un gate** — PIPELINE.md §0 lo richiede
   esplicitamente: un gate rosso si corregge oppure si documenta qui, mai si ignora.

---

## Formato

```
### <data> — <titolo>
**Contesto:** cosa ha reso necessaria la decisione
**Scelta:** cosa e stato deciso
**Alternative scartate:** e perche
**Reversibilita:** quanto costa tornare indietro
```

---

### 2026-08-07 — Python 3.11 invece di 3.9
**Contesto:** sul sistema sono installate entrambe le versioni.
**Scelta:** 3.11 per il venv del progetto.
**Alternative scartate:** 3.9 — le wheel PyTorch CUDA recenti non la coprono piu, e
l'interprete e piu lento; non irrilevante visto che il collo di bottiglia dell'MCTS
sara python-chess in puro Python (criticita #6).
**Reversibilita:** alta, basta ricreare il venv.

### 2026-08-07 — torch da indice cu126 invece di cu124
**Contesto:** il driver installato (566.14) supporta CUDA 12.6.
**Scelta:** installare da `--index-url .../cu126`.
**Alternative scartate:** cu124, il cui indice e fermo a torch 2.6 mentre cu126 arriva
a 2.13. Nessun motivo di restare indietro di sette minor su una GPU Ampere supportata
da entrambe.
**Reversibilita:** alta.

### 2026-08-07 — Distillazione da Stockfish esclusa
**Contesto:** darebbe +300-500 Elo per una notte di CPU.
**Scelta:** esclusa, come da piano.
**Alternative scartate:** distillare — trasferisce per copia la conoscenza di un altro
motore invece di produrne di nuova, e l'obiettivo del progetto e didattico.
**Conseguenza accettata:** la value head restera rumorosa; e il rischio strutturale del
progetto, da misurare al Gate 5 con la diagnosi §4.4.
**Reversibilita:** media — resta l'ultima risorsa se la diagnosi §4.4 va male.

### 2026-08-07 — Le valutazioni `[%eval]` dei PGN Lichess si tengono
**Contesto:** ~6% dei PGN Lichess contiene gia valutazioni Stockfish.
**Scelta:** tenerle come target ausiliario per la value head.
**Perche non contraddice la decisione precedente:** sono etichette che arrivano gratis
nei dati scaricati comunque, non una campagna di annotazione. Avendo escluso la
distillazione, sono l'unico correttivo disponibile per la componente debole.
**Reversibilita:** alta, e un peso di loss in `configs/default.yaml`.

### 2026-08-07 — Progetto cross-platform Windows + macOS
**Contesto:** il progetto nasce su Windows con RTX 3050, ma un collaboratore lavora su Mac.
**Scelta:** rendere `check.py` e `preflight.py` cross-platform, con i criteri specifici
della piattaforma che diventano **SKIP con motivazione** invece di FAIL. Aggiunto lo
Stadio 0-bis alla pipeline come prossimo passo assegnato a chi sta su Mac.
**Alternative scartate:**
- Ignorare il Mac e sviluppare solo su Windows — perde un collaboratore e, soprattutto,
  perde un test gratuito: due piattaforme diverse verificano l'assenza di dipendenze
  dalla piattaforma dentro l'encoder, cosa che una sola macchina non può fare.
- Rimuovere del tutto i check CUDA per farli passare ovunque — cancellerebbe un controllo
  reale sulla macchina dove conta. Un criterio inapplicabile va dichiarato tale, non tolto.
**Conseguenza:** i file golden dello Stadio 1 diventano anche un test di portabilità: se
divergono fra le due macchine, c'è un bug dipendente dalla piattaforma.
**Reversibilità:** alta.

### 2026-08-07 — Cython/Rust fuori dall'ambiente iniziale
**Contesto:** §4.3 ipotizza di riscrivere la generazione mosse.
**Scelta:** non includere `cython`/`maturin` nel venv adesso.
**Motivo:** regola #4 — niente ottimizzazione senza profiling. Averli installati e un
invito a ottimizzare prima di aver misurato.
**Reversibilita:** alta, si aggiungono quando il profiler lo conferma.

### 2026-08-08 — La baseline non impara i finali elementari
**Contesto:** con KR vs K e KQ vs K la baseline spinge il re avversario al bordo ma non
chiude, e arriva alla regola delle 50 mosse. Con materiale abbondante (KRR vs K) da matto.
**Scelta:** accettare il limite e registrarlo, invece di aggiungere tabelle di finale o
aumentare la profondita.
**Motivo:** §Fase 1 fissa l'obiettivo a 1200-1400 Elo, e la baseline serve come *metro*,
non come motore forte. Il criterio del Gate 2 e "batte random", non "converte ogni
finale vinto". Il costo di renderla piu forte si paga due volte: tempo speso ora, e un
metro piu difficile da superare per la rete senza che questo dica nulla di utile.
**Alternative scartate:**
- Tablebase Syzygy per i finali a pochi pezzi — sposta il problema fuori dal motore e
  rende la baseline non piu confrontabile con la rete, che le tablebase non le avra.
- Profondita 5-6 nei finali — costa secondi per mossa e non risolve: il matto con re e
  torre richiede una tecnica, non due semimosse in piu.
**Conseguenza:** contro l'avversario casuale una quota di partite finisce patta invece
che vinta. Il numero esatto e in RESULTS.md, misurato e non stimato.
**Reversibilita:** alta, e un termine di valutazione isolato in `_mate_drive`.

### 2026-08-08 — Ripetizione trattata come patta alla seconda occorrenza
**Contesto:** la ricerca controllava `is_repetition(3)`, e la baseline ripeteva le mosse
in posizione vinta perche a profondita 3 tutte le alternative valevano uguale.
**Scelta:** trattare come patta gia la **seconda** occorrenza dentro l'albero, piu una
penalita alla radice sulle mosse che ripetono quando il punteggio e positivo.
**Motivo:** e la convenzione standard dei motori (Stockfish compresa). Se una linea porta
a ripetere una posizione gia vista, l'avversario puo sempre insistere fino alla terza:
valutarla come patta e corretto, non conservativo.
**Alternative scartate:**
- Solo la penalita alla radice — insufficiente, la ripetizione nasce dentro l'albero.
- Tabella di trasposizione con conteggio delle occorrenze — la soluzione giusta per un
  motore vero, sproporzionata per cento righe di baseline.
**Reversibilita:** alta, due condizioni in `search.py`.

### 2026-08-08 — Finestra alpha-beta piena alla radice, non stretta
**Contesto:** la radice di `search()` usava `(-INFINITY, -alpha)` come tutti i nodi
interni. I punteggi dei rami potati non sono valori veri ma limiti di cutoff, e la
radice li riusa per raccogliere le mosse di pari merito da cui `rng` estrae: mosse
perdenti finivano nel gruppo delle migliori. La baseline pattava contro l'avversario
casuale (14W-9L-77D su 100).
**Scelta:** finestra piena `(-INFINITY, INFINITY)` per ogni mossa della radice.
**Costo:** circa 2-3x nei nodi visitati alla radice. La potatura vera resta dentro
`_negamax`, dove i punteggi non vengono riusati.
**Alternative scartate:**
- Tenere la finestra stretta e prendere solo `best_moves[0]` — elimina il tie-breaking
  casuale, che pero serve: senza, due baseline identiche giocano sempre la stessa
  partita e un match non ha valore statistico.
- Ricerca a finestra nulla piu re-search sui miglioramenti (PVS) — e la soluzione
  corretta per un motore vero, sproporzionata per una baseline da cento righe e con un
  costo di complessita che non ripaga a profondita 3-4.
**Verificato:** `test_le_mosse_pari_merito_sono_davvero_equivalenti` fallisce se il bug
viene reintrodotto (provato).
**Reversibilita:** alta, una riga.
