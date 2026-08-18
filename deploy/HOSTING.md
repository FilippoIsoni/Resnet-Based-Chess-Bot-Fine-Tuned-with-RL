# Pubblicare il bot online

Guida operativa: dal codice sul portatile a un indirizzo che chiunque puo aprire.
Tutto gratuito, senza carta di credito.

Ci sono **due cose separate** da pubblicare, in questo ordine:

```
  1. il motore   -> Render        (Python, ha bisogno di un server)
  2. la UI       -> GitHub Pages  (file statici, li serve gia GitHub)
```

Il motore va per primo: la UI ha bisogno del suo indirizzo per essere costruita.

## Perche Render e non Hugging Face

La guida diceva Hugging Face Spaces. **Da luglio 2026 non e piu gratuito** per quello che
serve a noi: gli Space Docker e Gradio richiedono un piano PRO da 9 $/mese, e restano
gratuiti solo gli Space *Static*, che servono file e non eseguono Python.

Fra i piani gratuiti rimasti, quasi tutti si fermano a **512 MB di RAM**. Il backend con
PyTorch ne occupa **608**, quindi non parte proprio: va in errore di memoria prima ancora
di caricare i pesi. Per questo il primo passaggio della guida e convertire la rete in
ONNX, che porta il consumo a **150 MB** — con le stesse identiche risposte, verificate
automaticamente.

| | RAM | Carta | Sospensione |
|---|---|---|---|
| **Render** | 512 MB | no | dopo 15 min, risveglio ~1 min |
| Koyeb | 512 MB | no | variabile |
| Northflank | 1 GB | **si** | mai |
| HF Spaces Docker | 16 GB | no | 9 $/mese |

Render e Koyeb vanno entrambi bene. La guida usa Render perche il risveglio e piu rapido.

---

## 1. Convertire la rete in ONNX

```bash
python scripts/export_onnx.py
```

Produce `runs/onnx/chessbot.onnx` piu un `chessbot.onnx.data` di ~46 MB — **i due file
vanno sempre insieme**, il primo da solo e un guscio vuoto.

Lo script non si limita a convertire: confronta le risposte delle due versioni su cinque
posizioni diverse e si rifiuta di dichiarare valido un modello che diverge. La verifica
misurata e uno scarto di 5,1e-07 contro una soglia di 1e-3, con la stessa mossa preferita
ovunque.

Il backend usa il modello ONNX in automatico appena lo trova. Per accertarsene:

```bash
python -m uvicorn chessbot.api.app:app --port 8000
curl http://127.0.0.1:8000/health      # deve dire "model_loaded": true
```

---

## 2. Caricare i pesi

I 46 MB del modello non stanno in git — il gate `check_no_large_files` rifiuta `.onnx` per
estensione. Vanno pubblicati a parte, e il posto piu comodo e una **release GitHub**,
perche il repo ce l'hai gia.

1. sul repository -> **Releases** -> **Draft a new release**
2. tag: `v0.2-onnx`, titolo a piacere
3. trascina **entrambi** i file: `chessbot.onnx` e `chessbot.onnx.data`
4. **Publish release**

Prendi nota degli indirizzi di download, nella forma:

```
https://github.com/TUONOME/REPO/releases/download/v0.2-onnx/chessbot.onnx
https://github.com/TUONOME/REPO/releases/download/v0.2-onnx/chessbot.onnx.data
```

---

## 3. Creare il servizio su Render

Serve un account gratuito su https://render.com (registrazione con GitHub, nessuna carta).

1. **New** -> **Web Service** -> collega il repository
2. **Language**: `Python 3`
3. **Build Command**:
   ```
   pip install torch --index-url https://download.pytorch.org/whl/cpu --no-deps &&
   pip install -r requirements-serve.txt && pip install -e . --no-deps &&
   mkdir -p runs/onnx &&
   curl -sL -o runs/onnx/chessbot.onnx https://github.com/TUONOME/REPO/releases/download/v0.2-onnx/chessbot.onnx &&
   curl -sL -o runs/onnx/chessbot.onnx.data https://github.com/TUONOME/REPO/releases/download/v0.2-onnx/chessbot.onnx.data
   ```
   (tutto su una riga sola; adatta `TUONOME/REPO`)
4. **Start Command**:
   ```
   uvicorn chessbot.api.app:app --host 0.0.0.0 --port $PORT --workers 1
   ```
   `--workers 1` non e negoziabile: il rate limiter e il flag di occupato vivono in
   memoria e valgono solo dentro un processo.
5. **Instance Type**: `Free`
6. **Create Web Service**

> **Perche installare torch se serviamo ONNX?** Non serve. `requirements-serve.txt` non lo
> elenca e il backend non lo importa quando trova il `.onnx`. Se il build risulta troppo
> lento, togli pure la prima riga del Build Command: serve solo come rete di sicurezza nel
> caso il download del modello fallisca e si debba ripiegare sui checkpoint.

L'indirizzo sara `https://<nome-servizio>.onrender.com`. Segnatelo per il punto 5.

---

## 4. Verificare

Il primo build richiede qualche minuto. Quando lo stato e **Live**:

```bash
curl https://<nome-servizio>.onrender.com/health
```

Deve rispondere `"model_loaded": true`. Se dice `false`, il download del modello e fallito:
controlla i log del build e che gli URL della release siano giusti.

Poi una mossa vera:

```bash
curl -X POST https://<nome-servizio>.onrender.com/move \
  -H "Content-Type: application/json" \
  -d '{"fen":"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1","level":"medium"}'
```

**Cronometra i tre livelli** — e l'unico numero che qui non si puo prevedere. In locale con
ONNX sono 0,15 / 0,44 / 1,7 s; su hardware condiviso aspettati 2-4 volte tanto. Se `hard`
supera gli 8 secondi, abbassalo senza toccare il codice: Environment ->
`CHESSBOT_HARD_SIMULATIONS` = `400`. La leva esiste apposta.

---

## 5. Pubblicare la UI

Due impostazioni sul repository GitHub, una volta sola:

1. **Settings -> Pages -> Source**: `GitHub Actions` (non "Deploy from a branch")
2. **Settings -> Secrets and variables -> Actions -> Variables -> New variable**:
   - nome: `BACKEND_URL`
   - valore: `https://<nome-servizio>.onrender.com`  (con `https://`, senza `/` finale)

Poi: scheda **Actions** -> workflow **pages** -> **Run workflow**. Da qui in avanti riparte
da solo ad ogni modifica dentro `web/`.

Il sito sara su `https://TUONOME.github.io/Resnet-Based-Chess-Bot-Fine-Tuned-with-RL/`.

Se `BACKEND_URL` manca o non inizia per `https://`, il workflow si ferma con un messaggio
esplicito invece di pubblicare un sito che sembra funzionare e non gioca.

---

## Cosa aspettarsi, e cosa non e un guasto

**Il primo caricamento dopo un quarto d'ora e lento.** Il servizio gratuito si sospende
dopo 15 minuti di inattivita e il risveglio richiede circa un minuto. La UI chiama
`/health` all'apertura della pagina proprio per far partire il risveglio mentre stai
scegliendo la difficolta, e la prima richiesta ha un timeout di 45 secondi invece di 15.
Non e rotto: sta tornando su.

**Regge poche partite insieme.** Le ricerche sono serializzate: una seconda partita in
corso riceve `503` e la UI dice di riprovare. Per un progetto da portfolio e abbondante;
non e dimensionato per traffico vero.

**Le 750 ore gratuite al mese** di Render bastano per un servizio solo, che comunque resta
sospeso quando nessuno gioca.

**La pagina bianca ha quasi sempre una causa sola.** Se il sito si carica vuoto, e il
`--base-href`: Pages serve da una sottocartella e Flutter senza quel parametro genera
percorsi assoluti. Il workflow lo ricava dal nome del repository, quindi non dovrebbe
succedere — ma se rinomini il repository, l'indirizzo cambia.

**Se la scacchiera si vede ma nessuna mossa arriva**, apri la console del browser (F12).
Un errore CORS significa che l'origine di Pages non e fra quelle autorizzate in
`src/chessbot/api/app.py` (`ALLOWED_ORIGINS`): va aggiunta li e ripubblicato il backend.
Nessun errore visibile ma richiesta bloccata significa quasi sempre mixed content, cioe un
`BACKEND_URL` scritto `http://` invece di `https://`.

---

## Riepilogo

| | Dove | Indirizzo |
|---|---|---|
| Modello ONNX | release GitHub | `github.com/TUONOME/REPO/releases/tag/v0.2-onnx` |
| Motore | Render | `<nome-servizio>.onrender.com` |
| UI | GitHub Pages | `TUONOME.github.io/<repo>/` |

Il contratto fra le ultime due e `docs/API_CONTRACT.md`: se cambia una delle due meta, si
aggiorna prima quel file.
