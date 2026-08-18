---
title: Chessbot API
emoji: ♟️
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# Chessbot — backend

API FastAPI per il motore ResNet + MCTS-PUCT (torch CPU). Nessuna UI qui: la UI
gira su GitHub Pages e parla con questo servizio secondo il contratto in
`docs/API_CONTRACT.md` del repository principale.

    curl https://<utente>-<spazio>.hf.space/health

    curl -X POST https://<utente>-<spazio>.hf.space/move \
      -H "Content-Type: application/json" \
      -d '{"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "level": "medium"}'

Cold start atteso ~30 s dopo un periodo di inattivita (Space gratuito) — la UI
chiama `/health` all'apertura della pagina proprio per farlo partire prima.

## Nota per chi pubblica questo Space

Questo file e `Dockerfile` vivono in `deploy/` dentro il monorepo di
sviluppo, come staging di cio che deve arrivare alla radice del repository
Git *separato* dello Space (Hugging Face richiede `Dockerfile` e `README.md`
con questo front-matter alla radice). Al momento della pubblicazione:

1. copiare `deploy/Dockerfile` e `deploy/README.md` alla radice del repo dello Space
2. il `Dockerfile` si aspetta comunque di essere costruito con la radice del
   *monorepo* come contesto (per `src/`, `requirements-serve.txt`, `runs/`) —
   se lo Space usa il proprio repository come contesto, sincronizzare anche
   `pyproject.toml`, `requirements-serve.txt`, `src/` e (se presente)
   `runs/` in quel repository, oppure configurare lo Space per puntare a
   questo monorepo come sorgente
