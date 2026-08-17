# Backend chessbot — Stadio 7 (docs/API_CONTRACT.md, docs/DECISIONS.md 2026-08-17).
#
# Contesto di build = RADICE del repository (non deploy/):
#   docker build -f deploy/Dockerfile -t chessbot-api .
#
# torch CPU invece di ONNX — riusa invariati Evaluator/run_mcts del motore.
# Pensato per Hugging Face Spaces (SDK Docker): porta fissa 7860, un solo
# worker Uvicorn (EngineState.busy e il rate limiter sono in-memory,
# condivisi solo dentro un unico processo).

FROM python:3.11-slim

WORKDIR /app

# torch CPU dall'indice dedicato — stessa riga gia usata in
# .github/workflows/checks.yml per evitare la wheel CUDA (enorme e inutile
# qui, lo Space non ha GPU).
COPY requirements-serve.txt /app/requirements-serve.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-serve.txt

# --no-deps: le dipendenze di serving sono gia risolte sopra. pyproject.toml
# dichiara le dipendenze dinamiche da requirements.txt (training/dati, non
# serving) — senza --no-deps qui si installerebbero scipy/tensorboard/
# zstandard inutilmente.
COPY pyproject.toml /app/pyproject.toml
COPY src/ /app/src/
RUN pip install --no-cache-dir -e . --no-deps

# runs/.gitkeep e tracciato in git nonostante runs/* sia in .gitignore, quindi
# questa directory esiste sempre nel contesto di build. Se un checkpoint reale
# e presente localmente al momento del build finisce nell'immagine; altrimenti
# resta vuota e il fallback grazioso di chessbot.api.engine.load_model()
# gestisce l'assenza senza errori (model_loaded: false, non un crash).
COPY runs/ /app/runs/

ENV CHESSBOT_RL_CHECKPOINT=/app/runs/rl/main/state.pt \
    CHESSBOT_SUPERVISED_CHECKPOINT=/app/runs/supervised/best.pt \
    CHESSBOT_DEVICE=cpu \
    CHESSBOT_HARD_SIMULATIONS=800

# Richiesta da HF Spaces SDK Docker: il container deve ascoltare esattamente
# sulla porta dichiarata in app_port nel front-matter di deploy/README.md.
EXPOSE 7860

RUN useradd -m -u 1000 appuser
USER appuser

CMD ["uvicorn", "chessbot.api.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
