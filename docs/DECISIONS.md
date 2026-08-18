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

### 2026-08-09 — Deduplica sulla posizione, non sulla FEN completa
**Contesto:** verificando il primo dataset da 20M posizioni ho trovato ~0.17% di
posizioni presenti in due split diversi. La deduplica confrontava le FEN complete, che
includono halfmove clock e numero di mossa: la stessa posizione raggiunta per
trasposizioni diverse ha contatori diversi, quindi FEN diverse, e passava il controllo.
**Scelta:** deduplica e capping usano `position_key()` — pezzi, tratto, arrocco, en
passant. I due campi contatore sono esclusi.
**Motivo:** per la rete due posizioni con gli stessi pezzi e lo stesso tratto sono la
stessa cosa: l'encoder codifica l'halfmove clock come piano, ma la posizione che la
policy deve valutare e identica. Tenerle in split diversi e leakage, punto.
**Alternative scartate:**
- Lasciare la FEN completa e accettare lo 0.17% — e piccolo, ma la criticita #3 esiste
  proprio perche il leakage non si vede: si manifesta come validation accuracy
  ottimistica, cioe come la metrica su cui si prenderanno tutte le decisioni dello
  Stadio 4. Un criterio del gate dice 0, non "quasi 0".
- Includere anche il conteggio ripetizioni nella chiave — non e nella FEN e andrebbe
  ricalcolato; l'effetto sul leakage sarebbe nullo.
**Conseguenza:** il dataset e stato rigenerato da zero (~2h). Le posizioni scartate per
deduplica passano da 326k a un valore piu alto: e il punto.
**Reversibilita:** alta, una funzione di quattro righe — ma cambiarla invalida il
dataset, che va rigenerato.

### 2026-08-13 — Rete a 12M parametri invece dei ~3,5M stimati dal piano
**Contesto:** §3.1 descrive l'architettura e stima ~3,5M parametri. Implementandola
esattamente come specificata ne vengono **12.029.715**. La stima non torna con
l'architettura che il piano stesso descrive: il solo `Linear(32*8*8 -> 4672)` della
policy head vale 9.568.256 pesi, e da solo supera il totale dichiarato.

    stem              22.144   0,2%
    8 blocchi      2.363.392  19,6%
    policy head    9.577.088  79,6%   <- il Linear finale
    value head        67.091   0,6%

**Scelta:** tenere l'architettura come specificata, 12M parametri.
**Motivo:** e quella scritta nel piano, e sta comodamente nei 6 GB della 3050 — il
training completo ha usato meno di 2 GB di VRAM e ha girato a 5.600 posizioni/s. La
stima sbagliata non ha conseguenze pratiche.
**Alternativa nota, se un domani servisse comprimere:** la policy head "convoluzionale
pura" di AlphaZero — Conv 1x1 da 128 a 73 canali, il cui output 73x8x8 = 4672 e gia lo
spazio delle mosse, senza alcun Linear. Costerebbe 9.344 parametri invece di 9,57M,
portando la rete a ~2,5M. Non e stata adottata perche non c'e un problema da risolvere:
la memoria non e un vincolo e il tempo di training e accettabile.
**Reversibilita:** media — cambiare la policy head invalida i checkpoint esistenti e
richiede di riallenare.

### 2026-08-13 — Label smoothing applicato solo alle mosse legali
**Contesto:** `F.cross_entropy(label_smoothing=0.05)` distribuisce massa su tutte le 4672
classi, comprese le ~4642 illegali portate a -1e9 dalla mascheratura. La loss valeva
49.683.864 invece di 3,40.
**Scelta:** `masked_cross_entropy()` in `training/loss.py`, che smootha sulla sola
distribuzione legale.
**Motivo:** il gradiente sui logit legali restava corretto anche col bug — la rete
imparava lo stesso — ma la loss era illeggibile e schiacciava il termine WDL di sette
ordini di grandezza, rendendo prive di senso tutte le soglie numeriche del Gate 4.
**Alternative scartate:**
- Togliere il label smoothing — e nel piano (§3.3) e serve: la mossa giocata da un umano
  non e l'unica ragionevole, e pretendere probabilita 1 su quella e un target sbagliato.
- Mascherare dopo la loss invece che prima — cambierebbe la distribuzione su cui la rete
  e ottimizzata rispetto a quella usata in inferenza (criticita #12).
**Reversibilita:** alta, una funzione isolata.

### 2026-08-17 — Stadio 7 diviso fra due persone, con un contratto scritto
**Contesto:** il deployment web ha due meta tecnologicamente distanti — una UI Flutter e
un server Python — e due persone su due sistemi operativi diversi. Svilupparle in
sequenza significa che una delle due aspetta.
**Scelta:** frontend a Filippo (Windows), backend al collaboratore su Mac, sviluppo in
parallelo, con `docs/API_CONTRACT.md` come unico punto di accordo vincolante. La UI si
costruisce contro un `FakeEngine` che restituisce mosse legali casuali, cosi puo essere
finita e pubblicata prima che il backend esista.
**Motivo:** l'unico accoppiamento reale fra le due meta e la forma delle richieste e delle
risposte. Fissarla per iscritto costa mezz'ora e rimuove la dipendenza temporale; lasciarla
implicita la fa scoprire durante il collegamento, quando entrambe le parti sono gia scritte
e ogni divergenza costa una modifica per lato.
**Alternative scartate:** generare il client Dart da OpenAPI — eliminerebbe le divergenze
di forma ma richiede che il backend esista prima della UI, cioe reintroduce esattamente la
dipendenza che si vuole togliere.
**Reversibilita:** alta.

### 2026-08-17 — torch CPU invece di ONNX per il backend
**Contesto:** l'Appendice B prescrive di esportare la rete in ONNX e servirla con
onnxruntime (~15 MB) invece che con PyTorch (200+ MB), e `requirements-serve.txt` era
scritto di conseguenza.
**Scelta:** servire con torch CPU, riusando `Evaluator` e `run_mcts` invariati.
**Motivo:** su Hugging Face Spaces il peso dell'immagine non e un vincolo, e l'export
introdurrebbe un rischio di divergenza numerica su una rete la cui value head e gia il
punto debole del progetto (criticita #5). Inoltre `.onnx` e vietato sia da `.gitignore`
sia da `forbidden_ext` in `scripts/check.py`, quindi il vantaggio "artefatto leggero e
committabile" non esiste comunque. Da notare che `onnx` e `onnxscript` non sono nemmeno
installati nel venv: l'export non e mai stato provato.
**Alternative scartate:** esportare in ONNX e scrivere un evaluator alternativo con la
stessa interfaccia — fattibile, e la classe `Evaluator` e progettata per essere sostituita,
ma richiede l'export piu la verifica di parita a 1e-3 che il piano stesso impone. Lavoro
reale per un beneficio che su questo hosting non si materializza.
**Reversibilita:** media — `Evaluator` e sostituibile per costruzione, ma va scritto
l'export e la batteria di parita.

### 2026-08-17 — Hugging Face Spaces invece di Fly.io
**Contesto:** l'Appendice B raccomanda Fly.io per il risveglio rapido (2-5 s).
**Scelta:** Hugging Face Spaces con SDK Docker.
**Motivo:** gratuito senza carta di credito, HTTPS incluso, ed e l'ecosistema naturale per
un modello ML. Fly.io richiede una carta anche restando nel tier gratuito.
**Conseguenza accettata:** cold start di ~30 s invece di 2-5. Va mascherato nella UI con
tre accorgimenti: ping a `/health` all'apertura della pagina (il risveglio parte mentre
l'utente sceglie la difficolta), stato "si sta svegliando" esplicito oltre i 3 s, e
possibilita di muovere subito perche le regole girano lato client. Il timeout del client
va a 45 s sulla prima richiesta: il valore unico di 10 s suggerito dal piano fallirebbe
sistematicamente al risveglio.
**Reversibilita:** alta — cambia il Dockerfile e un URL.

### 2026-08-17 — Nessun riuso dell'albero MCTS fra richieste
**Contesto:** l'Appendice B prevede che il campo `session` abiliti il riuso dell'albero via
cache LRU, stimando -30-40% di simulazioni.
**Scelta:** non implementato. Il campo resta nello schema ma serve solo per rate limit e
log.
**Motivo:** `run_mcts` costruisce sempre una radice nuova e non accetta un parametro
`root=`. Aggiungerlo significa modificare il cuore della ricerca che ha appena superato il
Gate 5 e prodotto i 1964 Elo misurati. Il beneficio a 200 simulazioni e circa 0,15 s su
0,38 — impercettibile per un umano — e lo Space si addormenta dopo inattivita, azzerando
ogni cache proprio nel caso "utente che torna dopo la pausa", che e quello in cui servirebbe.
**Alternative scartate:** implementarlo comunque per aderenza al piano. Il piano stesso
dice di non ottimizzare senza misurare (regola permanente #4).
**Reversibilita:** alta — il punto d'innesto e un parametro `root: Node | None = None` in
`run_mcts`, se un domani i tempi lo giustificassero.

### 2026-08-18 — Marcia indietro su ONNX: si torna a quanto prescriveva il piano
**Contesto:** il 2026-08-17 avevamo scartato ONNX per servire con torch, perche
su Hugging Face Spaces il peso dell'immagine non era un vincolo. Quella premessa
e caduta: da luglio 2026 HF richiede un piano PRO (9 $/mese) per gli Space
Docker e Gradio, e restano gratuiti solo gli Space *Static*, che non eseguono
Python.
**Misura che ha deciso:** il backend con torch occupa **608 MB** di RSS, di cui
489 solo per `import torch`. Praticamente ogni piano gratuito rimasto si ferma a
512 MB (Render, Koyeb, SnapDeploy), quindi il servizio non parte proprio.
Servito da onnxruntime lo stesso motore occupa **150 MB**: quattro volte meno, e
il problema di hosting sparisce senza pagare nulla.
**Scelta:** export ONNX come percorso primario (`scripts/export_onnx.py`,
`api/onnx_evaluator.py`), con ripiego automatico su torch quando il `.onnx` non
c'e — il caso normale in sviluppo, dato che e un artefatto di build gitignored.
**Parita verificata**, come il piano chiede: scarto massimo **5,1e-07** contro
una soglia di 1e-3, e mossa preferita identica su tutte le posizioni di prova.
Il controllo e automatizzato in `tests/unit/test_onnx_parity.py`, non fatto una
volta a mano: un export verificato a mano e un export che diverge in silenzio
alla prossima modifica della rete.
**Conseguenza non prevista:** ridurre la memoria ha richiesto di rendere pigro
l'import di torch anche in `search/mcts.py`, dove serviva solo a `Evaluator` ma
veniva pagato da chiunque importasse `chessbot.search`. Il Gate 5 resta verde
(stessi +486 ± 103 Elo), quindi il refactor non ha toccato il comportamento.
**Alternative scartate:** pagare i 9 $/mese (funziona, ma il progetto e
didattico e la soluzione gratuita esisteva); Northflank, unico gratuito con 1 GB
e senza sospensione, scartato perche chiede comunque la carta per la verifica.
**Reversibilita:** alta — basta non generare il `.onnx` e il backend riprende a
servire con torch, senza modifiche al codice.

### 2026-08-18 — Cache di mypy disattivata nel gate
**Contesto:** mypy 1.18.2 crasha con `NotImplementedError: Cannot serialize
TypeGuardedType instance` mentre **scrive** la cache, su qualunque modulo che
importi torch. Il gate L1 risultava rosso senza dire niente sul codice: un
guasto dello strumento travestito da errore del progetto. Il crash precede le
modifiche di oggi — verificato facendo stash e rilanciando su main pulito.
**Scelta:** `check_mypy()` passa `--cache-dir=os.devnull`.
**Motivo:** e l'unica cosa che funziona. `--no-incremental` e
`incremental = false` lasciano comunque scrivere la cache, e cosi una directory
temporanea vera. Solo un percorso non scrivibile la impedisce, e a quel punto
mypy prosegue e riporta i risultati normalmente.
`os.devnull` e non `/dev/null` a mano: quest'ultimo funziona da Git Bash, che lo
traduce, ma non da PowerShell.
**Cosa si perde:** solo il riuso fra esecuzioni, una decina di secondi in piu su
`src`. Nessun controllo in meno — e la stessa analisi.
**Nota:** il crash mascherava **3 errori di tipo veri** in `onnx_evaluator.py`
(una variabile riusata con due tipi diversi), corretti. Il costo di un gate
rotto non e il tempo perso a ripararlo: e quello che nasconde mentre e rotto.
**Reversibilita:** alta, da togliere quando mypy corregge il bug.
