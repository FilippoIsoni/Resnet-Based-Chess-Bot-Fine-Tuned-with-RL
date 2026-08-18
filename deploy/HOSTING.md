# Pubblicare il bot online

Guida operativa: dal codice sul portatile a un indirizzo che chiunque puo aprire.

Ci sono **due cose separate** da pubblicare, in questo ordine:

```
  1. il motore   -> Hugging Face Spaces   (Python, ha bisogno di un server)
  2. la UI       -> GitHub Pages          (file statici, li serve gia GitHub)
```

Il motore va per primo: la UI ha bisogno del suo indirizzo per essere costruita.

I pesi (`runs/rl/main/state.pt`, 48 MB) non sono in git e non possono esserci — il gate
`check_no_large_files` li rifiuta. Vanno caricati a parte, ed e il passaggio che la
maggior parte delle guide salta. E il punto 2 qui sotto.

---

## 1. Account Hugging Face

Serve solo un account gratuito, senza carta di credito: https://huggingface.co/join

L'unica cosa che conta e il **nome utente**, perche finisce nell'indirizzo del servizio.
Nel seguito lo chiamo `TUONOME`.

---

## 2. Caricare i pesi

I pesi vivono in un **repository modello**, separato dallo Space. E il modo previsto
dalla piattaforma: un repo modello e pensato per file grossi, uno Space per il codice.

Dal browser:

1. https://huggingface.co/new — sezione **Model** (non Space, non Dataset)
2. nome: `chessbot-rl`, visibilita **Public**
3. una volta creato: **Files** -> **Add file** -> **Upload files**
4. trascina `runs/rl/main/state.pt` e conferma con **Commit changes**

Sono 48 MB: un paio di minuti con una connessione normale.

Il repo ora e `TUONOME/chessbot-rl` e il file e scaricabile senza token, perche pubblico.

> **Se preferisci tenerlo privato** funziona lo stesso, ma serve un token: Settings ->
> Access Tokens -> New token (permesso *read*), da mettere poi fra i Secrets dello Space
> come `HF_TOKEN`. Pubblico e piu semplice e non espone nulla di sensibile: sono pesi di
> una rete, non dati.

---

## 3. Creare lo Space

1. https://huggingface.co/new-space
2. **Space name**: `chessbot-api`
3. **License**: quella che preferisci (MIT va bene)
4. **SDK**: **Docker** -> *Blank* (non Gradio, non Streamlit)
5. **Hardware**: *CPU basic* — gratuito, 2 vCPU, 16 GB
6. **Public**
7. **Create Space**

L'indirizzo sara `https://TUONOME-chessbot-api.hf.space`. Segnatelo: serve al punto 6.

---

## 4. Caricare il backend nello Space

Uno Space e un repository git. Il modo piu semplice e clonarlo e copiarci i file:

```bash
# fuori dalla cartella del progetto, per non annidare due repo
cd ~
git clone https://huggingface.co/spaces/TUONOME/chessbot-api
cd chessbot-api
```

Al primo push HF chiede le credenziali: nome utente e un **token con permesso write**
(Settings -> Access Tokens), non la password dell'account.

Ora copia dal progetto quello che serve al server. Da `chessbot-api/`:

```bash
PROG=~/Desktop/Chess-bot/chess-bot     # adatta il percorso

cp $PROG/deploy/Dockerfile          .
cp $PROG/deploy/README.md           .
cp $PROG/requirements-serve.txt     .
cp $PROG/pyproject.toml             .
cp -r $PROG/src                     .
```

**Il `Dockerfile` va adattato in un punto.** Quello nel progetto si aspetta i pesi in
`runs/`, che nello Space non esistono. Sostituisci la riga `COPY runs/ /app/runs/` e le
variabili d'ambiente sotto con:

```dockerfile
# I pesi arrivano dal repo modello al primo avvio. hf_hub_download li mette in
# cache su disco: al riavvio a caldo non li riscarica.
ENV HF_HOME=/tmp/hf \
    CHESSBOT_HF_REPO=TUONOME/chessbot-rl \
    CHESSBOT_DEVICE=cpu \
    CHESSBOT_HARD_SIMULATIONS=800
```

e aggiungi `huggingface_hub==0.36.0` a `requirements-serve.txt`.

Poi in `src/chessbot/api/engine.py`, nella funzione `load_model`, il percorso del
checkpoint va risolto scaricandolo quando non e sul disco:

```python
def _rl_checkpoint_path() -> Path:
    explicit = os.environ.get("CHESSBOT_RL_CHECKPOINT")
    if explicit:
        return Path(explicit)

    repo = os.environ.get("CHESSBOT_HF_REPO")
    if repo:
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download(repo_id=repo, filename="state.pt"))

    return DEFAULT_RL_CHECKPOINT
```

In locale niente cambia: `CHESSBOT_HF_REPO` non e impostata e vale il percorso di prima.

Infine:

```bash
git add -A
git commit -m "Backend chessbot"
git push
```

---

## 5. Verificare che funzioni

La scheda **Logs** dello Space mostra la build. La prima volta sono **10-15 minuti**:
scarica torch, che e grosso. Le successive sono piu rapide grazie alla cache.

Quando lo stato diventa **Running**:

```bash
curl https://TUONOME-chessbot-api.hf.space/health
```

Deve rispondere `"model_loaded": true`. Se dice `false`, i pesi non sono stati trovati:
controlla `CHESSBOT_HF_REPO` e che il repo modello sia pubblico.

Poi una mossa vera:

```bash
curl -X POST https://TUONOME-chessbot-api.hf.space/move \
  -H "Content-Type: application/json" \
  -d '{"fen":"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1","level":"medium"}'
```

**Cronometra i tre livelli** — e l'unico numero che qui non si puo prevedere. In locale
sono 0,2 / 0,6 / 2,3 s; su HF aspettati 2-4 volte tanto. Se `hard` supera gli 8 secondi,
abbassalo senza toccare il codice: Settings -> Variables -> `CHESSBOT_HARD_SIMULATIONS`
= `400`. La leva esiste apposta.

---

## 6. Pubblicare la UI

Due impostazioni sul repository GitHub, una volta sola:

1. **Settings -> Pages -> Source**: `GitHub Actions` (non "Deploy from a branch")
2. **Settings -> Secrets and variables -> Actions -> Variables -> New variable**:
   - nome: `BACKEND_URL`
   - valore: `https://TUONOME-chessbot-api.hf.space`  (con `https://`, senza `/` finale)

Poi lancia il deploy: scheda **Actions** -> workflow **pages** -> **Run workflow**. Da
qui in avanti riparte da solo ad ogni modifica dentro `web/`.

Il sito sara su `https://TUONOME.github.io/Resnet-Based-Chess-Bot-Fine-Tuned-with-RL/`.

Se `BACKEND_URL` manca o non inizia per `https://`, il workflow si ferma con un messaggio
esplicito invece di pubblicare un sito che sembra funzionare e non gioca.

---

## Cosa aspettarsi, e cosa non e un guasto

**Il primo caricamento dopo qualche ora e lento.** Lo Space gratuito si addormenta dopo un
periodo di inattivita e il risveglio richiede una trentina di secondi. La UI chiama
`/health` all'apertura della pagina proprio per far partire il risveglio mentre stai
scegliendo la difficolta, e la prima richiesta ha un timeout di 45 secondi invece di 15.
Non e rotto: sta tornando su.

**Regge poche partite insieme.** Le ricerche sono serializzate: una seconda partita in
corso riceve `503` e la UI dice di riprovare. Con 2 vCPU sono circa 5-8 giocatori
contemporanei a livello medio, 2-3 a difficile. Per un progetto da portfolio e
abbondante; non e dimensionato per traffico vero.

**La pagina bianca ha quasi sempre una causa sola.** Se il sito si carica vuoto, e il
`--base-href`: Pages serve da una sottocartella e Flutter senza quel parametro genera
percorsi assoluti. Il workflow lo ricava dal nome del repository, quindi non dovrebbe
succedere — ma se rinomini il repository, ricordati che l'indirizzo cambia.

**Se la scacchiera si vede ma nessuna mossa arriva**, apri la console del browser (F12).
Un errore CORS significa che l'origine di Pages non e fra quelle autorizzate in
`src/chessbot/api/app.py` (`ALLOWED_ORIGINS`): va aggiunta li e ripubblicato lo Space.
Nessun errore visibile ma richiesta bloccata significa quasi sempre mixed content, cioe
un `BACKEND_URL` scritto `http://` invece di `https://`.

---

## Riepilogo

| | Dove | Indirizzo |
|---|---|---|
| Pesi | repo modello HF | `huggingface.co/TUONOME/chessbot-rl` |
| Motore | HF Space | `TUONOME-chessbot-api.hf.space` |
| UI | GitHub Pages | `TUONOME.github.io/<repo>/` |

Il contratto fra le ultime due e `docs/API_CONTRACT.md`: se cambia una delle due meta,
si aggiorna prima quel file.
