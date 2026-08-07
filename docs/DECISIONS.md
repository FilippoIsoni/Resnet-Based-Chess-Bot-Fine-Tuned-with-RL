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
