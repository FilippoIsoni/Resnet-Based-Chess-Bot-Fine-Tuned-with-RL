# Contratto fra UI e backend

**Questo documento e l'unico punto di accordo fra le due meta del deployment.** La UI
Flutter e il backend Python vengono sviluppati da persone diverse, in parallelo, senza
vedersi girare a vicenda: finche entrambe le parti rispettano quanto scritto qui, il
collegamento funziona al primo tentativo.

Se una delle due parti ha bisogno di cambiare qualcosa, **si modifica prima questo file e
lo si comunica all'altra persona**, non si cambia il codice sperando che l'altro se ne
accorga.

| | Chi | Cosa |
|---|---|---|
| Backend | il collaboratore macOS | FastAPI + motore, deploy su Hugging Face Spaces |
| Frontend | Filippo | Flutter Web, deploy su GitHub Pages |

Contesto architetturale completo in `piano-motore-scacchi.md` Appendice B, con le tre
deviazioni registrate in `docs/DECISIONS.md`.

---

## Endpoint

Base URL di produzione: da definire quando lo Space esiste, nella forma
`https://<utente>-<nome-space>.hf.space`. In sviluppo: `http://127.0.0.1:8000`.

### `GET /health`

Serve a due cose: sapere se il backend e vivo, e **svegliarlo**. Lo Space gratuito si
addormenta dopo inattivita e il risveglio costa ~30 s; la UI chiama `/health` all'apertura
della pagina proprio per far partire il risveglio mentre l'utente legge il titolo.

Deve rispondere **anche mentre una ricerca e in corso**. Se il backend serializza le
ricerche con un semaforo, `/health` deve stare fuori da quel semaforo, altrimenti la UI
mostra "offline" mentre il server sta semplicemente lavorando.

Risposta `200`:

```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cpu",
  "uptime_s": 137.4
}
```

`model_loaded: false` e legittimo durante il caricamento dei pesi all'avvio: la UI lo
interpreta come "si sta svegliando", non come errore.

### `POST /move`

Richiesta:

```json
{
  "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
  "level": "medium",
  "session": "a3f9c1e2"
}
```

| Campo | Tipo | Obbligatorio | Note |
|---|---|---|---|
| `fen` | string | si | posizione completa in cui tocca al motore |
| `level` | `"easy"` \| `"medium"` \| `"hard"` | no, default `"medium"` | |
| `session` | string \| null | no | identificatore opaco generato dalla UI, stabile per partita |

Risposta `200`:

```json
{
  "move": "e2e4",
  "san": "e4",
  "eval": 0.12,
  "pv": ["e4", "e5", "Nf3"],
  "ms": 380,
  "sims": 200,
  "game_over": false,
  "result": null
}
```

| Campo | Tipo | Significato |
|---|---|---|
| `move` | string \| null | mossa scelta in UCI; `null` solo se la posizione e gia terminale |
| `san` | string \| null | la stessa mossa in notazione algebrica, per la cronologia |
| `eval` | float in `[-1, 1]` | vedi sotto — **attenzione al punto di vista** |
| `pv` | string[] | linea principale in SAN, fino a 10 mosse, puo essere vuota |
| `ms` | int | millisecondi impiegati dalla ricerca |
| `sims` | int | simulazioni effettivamente eseguite |
| `game_over` | bool | vero se la partita e finita **dopo** la mossa restituita |
| `result` | string \| null | `"1-0"`, `"0-1"`, `"1/2-1/2"`, altrimenti `null` |

---

## Le due cose che e facile sbagliare

### 1. Il punto di vista di `eval`

**`eval` e dal punto di vista di chi deve muovere nella posizione inviata.** Positivo
significa "chi muove sta meglio", non "il bianco sta meglio".

E la convenzione negamax usata in tutto il motore (criticita #2, documentata in
`src/chessbot/search/node.py`): il backup MCTS nega il valore ad ogni livello, e
`encode_board` orienta sempre la scacchiera dal lato che muove.

**Conseguenza per la UI:** una barra di valutazione vuole il punto di vista del bianco,
fisso. Se si mostra `eval` cosi com'e, la barra salta da un lato all'altro ad ogni mossa.
La conversione (`if (nero muove) eval = -eval`) va fatta **in un solo posto** nel client
Dart, subito dopo il parsing della risposta.

Questo attraversa tre confini — motore, JSON, UI — e ognuno puo invertire il segno senza
che nulla crashi. Entrambe le parti hanno un test sul segno.

### 2. `move` puo essere `null`

Succede se il FEN inviato descrive gia una posizione terminale (matto, stallo). La UI non
dovrebbe mai chiedere una mossa in quel caso, ma il backend lo gestisce comunque e la UI
non deve crashare se lo riceve.

---

## Errori

| Codice | Quando | Corpo |
|---|---|---|
| `422` | FEN malformato, FEN sintatticamente valido ma posizione illegale, `level` fuori dominio | formato standard FastAPI |
| `429` | rate limit superato (~30 richieste/minuto per IP) | `{"detail": "..."}` |
| `503` | server occupato, coda piena | `{"detail": "..."}` + header `Retry-After` |
| `500` | errore interno | `{"detail": "..."}` senza stacktrace |

La UI mostra un messaggio comprensibile per ciascuno, non il JSON grezzo. Su `503` puo
riprovare una volta dopo il ritardo indicato.

---

## CORS

Il backend deve autorizzare esplicitamente l'origine di GitHub Pages:

```
https://filippoisoni.github.io
```

piu `http://localhost:*` per lo sviluppo. **Non `["*"]`**: con `allow_credentials` non
funziona, ed e la trappola documentata nell'Appendice B.

Se l'header manca, il browser blocca la risposta e la UI vede un errore di rete generico,
senza spiegazione utile in console.

## HTTPS

GitHub Pages serve su HTTPS. Se il backend fosse su `http://`, il browser blocca la
richiesta come mixed content **senza errore visibile all'utente** — la UI sembra
semplicemente rotta. Hugging Face Spaces fornisce HTTPS, quindi il problema non si pone,
ma l'URL configurato nella UI va scritto `https://`.

---

## I tre livelli

La difficolta e il numero di simulazioni MCTS: stessa rete, profondita di ricerca diversa.

| Livello | Simulazioni | Forza |
|---|---|---|
| `easy` | 50 | ~1500 Elo (stimato) |
| `medium` | 200 | **1964 +/- 62 Elo (misurato)** |
| `hard` | 800 | ~2100 Elo (stimato) |

Solo `medium` e misurato, contro Stockfish a forza limitata (`scripts/run_elo_ladder.py`).
Gli altri due sono estrapolati dalla curva di monotonicita del Gate 5 e vanno presentati
come stime — regola #1: un numero senza intervallo di confidenza non e una misura.

Il livello `easy` usa `temperature=0.8` nella ricerca: campiona fra le mosse ben visitate
invece di prendere sempre la piu visitata, cosi non ripete la stessa partita. Non e un
indebolimento — quello lo fanno gia le 50 simulazioni — e varieta.

**Se `hard` risulta troppo lento sull'hardware gratuito** (oltre ~8 s per mossa), si
abbassa a 400 simulazioni. E una decisione del backend, da prendere su tempi misurati; va
comunicata ma non richiede modifiche alla UI, che manda solo l'etichetta.

## Prestazioni attese

Misurate in locale su CPU (Windows, RTX 3050 non usata): 50 sim = 0.12 s, 200 sim = 0.38 s,
800 sim = 1.21 s. Su Hugging Face Spaces free (2 vCPU) aspettarsi **2-4x piu lento**.

Con la ricerca serializzata regge circa 5-8 utenti simultanei a livello medio, 2-3 a
livello difficile. Non e dimensionato per traffico reale ed e giusto cosi.

---

## Come lavorare in parallelo senza bloccarsi

**Il frontend non aspetta il backend.** La UI implementa un'interfaccia `Engine` con due
realizzazioni: `FakeEngine`, che restituisce una mossa legale casuale con valutazione
finta, e `HttpEngine`, che parla con questo contratto. Tutta la UI si costruisce e si
prova contro `FakeEngine`.

**Il backend non aspetta il frontend.** Si verifica con `curl` e con i test pytest.

Il collegamento vero avviene quando entrambe le parti sono verdi separatamente. Se il
contratto e stato rispettato, e questione di cambiare un URL.
