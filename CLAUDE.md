# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

AlphaZero-style chess engine in Python 3.11 + PyTorch: ResNet policy/value network + MCTS (PUCT), trained via supervised learning on Lichess games then fine-tuned with Expert Iteration (self-play RL). Distillation from Stockfish is deliberately excluded — the goal is didactic (build a strong engine from scratch), not maximum strength.

Two docs are the source of truth and should be read before major changes:
- `piano-motore-scacchi.md` — architectural/technical plan (the *what*)
- `PIPELINE.md` — operational workflow: stages, gates, permanent rules (the *how*)

Also check `docs/DECISIONS.md`, `docs/JOURNAL.md`, `docs/RESULTS.md` for accumulated decisions/history, and `docs/CHECKLIST.md`.

## Current state (2026-08-14)

**Gates 1-5 are green. The bot works.** Next stage is 6 (Expert Iteration).

| Gate | Status | Headline result |
|---|---|---|
| 0 setup | green | — |
| 0-bis macOS | **open** | assigned to the macOS collaborator; golden files must be bit-identical across both machines |
| 1 encoding | green | 9 criteria; move round-trip on 1.78M legal moves, 0 failures |
| 2 baseline | green | 5 criteria; **100W-0L-0D vs random** over 100 games (recorded in `runs/gate2/`) |
| 3 data | green | 10 criteria; see dataset below |
| 4 train | green | 7 criteria; **top-1 51.66%** on validation (peak 52.11%) |
| 5 mcts | green | 9 criteria; **+486 Elo ± 103** for MCTS over raw policy |
| 6 rl-entry | not started | — |

**The trained network exists.** `runs/supervised/best.pt` (144 MB, gitignored) —
ResNet 8×128, 12.0M parameters, 12 epochs over 233.5M positions in 11.8 h on the RTX 3050.
Published as GitHub release `v0.1-supervised` so the macOS collaborator can use it without
retraining.

| | |
|---|---|
| Policy top-1 / top-5 | **51.66% / 89.37%** |
| Value head (WDL cross-entropy) | 0.9038 — barely above the 0.9710 of ignoring the position |
| MCTS speed | 1,276 sim/s with `batch_size_leaves=24` (213 without batching) |

**The value head is weak and it did not matter.** Criticità #5 materialised exactly as
predicted — without Stockfish distillation the game outcome is too noisy a label, and the
network gained only 0.067 nats over a strategy that ignores the board. The §4.4 diagnosis
was supposed to decide whether that made search useless: it gave **+486 Elo over 200
games**, three times the 150-Elo threshold. Search adds strength by exploring, not only by
evaluating — terminals are exact and need no network at all. **None of the §4.4 mitigations
were applied**, they remain available for Stage 6.

**The dataset exists and is verified.** `data/processed/` holds **20,000,000 positions**
(2.0 GB) from `lichess_db_standard_rated_2026-07.pgn.zst`, built at commit `669d452`:

| Split | Positions | Shards |
|---|---|---|
| train | 19,117,346 | 77 |
| val | 447,011 | 2 |
| test | 435,643 | 2 |

274,168 games kept out of 26,990,325 read (1.02% — the filters drop bullet/blitz and
sub-2000 Elo). 33.2% of positions carry a Stockfish `[%eval]` label. Three independent
verifications passed: round-trip on all 20M with 0 illegal moves, position↔move alignment
**100.0000%**, and move quality **8.6× better than random** on Stockfish top-1.

The dataset is **not in git** (2 GB) but is fully reproducible: `download_pgn.py` +
`build_dataset.py` with the same seed produce it byte-for-byte, and `manifest.json`
records the commit hash and config. `data/raw/` holds the 27 GB source dump.

Stockfish 18 (AVX2 build) is installed at `tools/stockfish/stockfish.exe`, gitignored —
macOS users need their own binary. It is required by `verify_move_quality.py` and by the
Elo ladder from Stage 2 onward.

What Stage 4 needs first: a DataLoader that expands the 105-byte packed rows into 19×8×8
tensors on the fly (`encoding/board.py::encode_board` is written and gate-tested but not
yet wired into the data path — nothing calls it outside the golden tests).

**Gate timing caveat.** Some criteria are slow and the gate runner samples them so that
`check.py --stage <name>` stays usable: Gate 3 reads 2 shards per split rather than all
20M rows, and its Stockfish check uses 60 positions. Gate 2's `perft profondo e match
brevi` runs baseline-vs-baseline games and takes **~10 minutes** on its own (verified:
`pytest tests/integration -m slow` → 8 passed in 602s) — under a short shell timeout it
gets killed mid-run and the gate reports FAIL for it. That FAIL is a timeout, not a
regression. The full-coverage numbers are the ones recorded in `docs/RESULTS.md`;
reproduce them by running the scripts directly without the sampling flags, allowing
10–25 minutes each.

## Commands

### Setup (Python 3.11 required, venv at `.venv`)
```bash
python3.11 -m venv .venv && source .venv/bin/activate   # macOS; see README for Windows
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu126   # or plain `pip install torch` on macOS (no CUDA wheels)
pip install -r requirements-dev.txt
pip install -e .
pre-commit install
python scripts/check.py --stage setup
```

### Gate runner (the project's task runner — use instead of ad hoc lint/test commands)
```bash
python scripts/check.py --level 1        # pre-commit, <20s: ruff lint+format, mypy, pytest -m fast, no-large-files
python scripts/check.py --level 2        # pre-merge: level 1 + full unit/integration/golden suite
python scripts/check.py --stage <name>   # phase gate: setup | encoding | baseline | data | train | mcts | rl-entry
python scripts/check.py --all
```
No stage advances until its gate is green (PIPELINE.md §0 — non-negotiable).

### Tests (pytest, config in `pyproject.toml`, `pythonpath = ["src"]`)
```bash
pytest                                     # full suite
pytest -m fast                             # what level 1 runs
pytest -m "not slow"                       # what level 2 runs
pytest tests/unit/test_encoding_moves.py   # single file
pytest tests/unit -k test_name             # single test by keyword
```
Markers: `fast`, `slow`, `golden`, `gpu`.

### Lint / format / types
```bash
ruff check src tests scripts
ruff format src tests scripts
mypy src
```

### Data pipeline & training scripts (`scripts/`)
```bash
python scripts/download_pgn.py --month YYYY-MM       # fetch Lichess PGN dump
python scripts/build_dataset.py --input <pgn.zst>    # PGN -> shards in data/processed/
python scripts/verify_dataset.py --data data/processed
python scripts/verify_alignment.py --data data/processed
python scripts/verify_move_quality.py --data data/processed   # needs tools/stockfish/
python scripts/gen_golden.py [--check]                # encoder golden files (must be bit-identical Win/macOS)
python scripts/run_gate2_match.py --games 100         # baseline vs random match
python scripts/preflight.py                           # run before any long GPU training session
```

## Architecture

`src/chessbot/` — implementation status by module (mirrors the 5 phases/stages in PIPELINE.md):

| Module | Status | Responsibility |
|---|---|---|
| `encoding/` | done | position → 19×8×8 tensor (`board.py`), move ↔ index [0,4672) AlphaZero 8×8×73 scheme (`moves.py`), stable cross-platform tensor hashing (`digest.py`) |
| `baseline/` | done | minimax/negamax+alphabeta w/ quiescence & MVV-LVA (`search.py`), material+PST eval (`evaluation.py`), perft (`perft.py`), match runner w/ confidence intervals (`match.py`) — sanity-check baseline (~1300 Elo) |
| `data/` | done | PGN parsing/streaming `.zst` (`pgn.py`), filters incl. Elo/time-control (`filters.py`), game-id-hash train/val/test split + FEN dedup (`split.py`), mmap'd binary shard storage (`storage.py`) |
| `model/` | stub | ResNet + policy head + WDL value head |
| `search/` | stub | MCTS PUCT, leaf batching w/ virtual loss |
| `training/` | stub | supervised loop + Expert Iteration loop (same loss, label source changes) |
| `eval/` | stub | match runner, Elo/SPRT, tactics suites |
| `api/` | stub | FastAPI deploy endpoint |
| `utils/` | partial | `seed.py` (global determinism); `configs/*.yaml` reference a pydantic schema at `utils/config.py` that does not exist yet |

Config: plain YAML validated by pydantic, not Hydra. `configs/default.yaml` is the single source of truth (paths, encoding, data, model, train, search, eval, logging); `configs/rl.yaml` extends it for the Expert Iteration phase (parallel self-play games, Dirichlet noise, gating/SPRT thresholds, hard stop criteria).

Stages gate progression (PIPELINE.md): 0 setup → 0-bis macOS/MPS parity → 1 encoding → 2 baseline → 3 data → 4 supervised train → 5 MCTS ("bot funzionante") → 6 RL/Expert Iteration.

## Critical invariants

These are correctness bugs that don't crash — they silently produce a weaker or wrong engine, which is why the gate system (`scripts/check.py --stage`) exists. Numbering follows PIPELINE.md/piano-motore-scacchi.md "criticità":

1. **Board mirroring must mirror the move index too.** The board is always encoded from the side-to-move's perspective (mirrored + colors swapped when Black to move). If the move label index isn't mirrored along with it, half the dataset trains on wrong labels with no error — only symptom is top-1 accuracy plateauing near 25% instead of ~50%. Both are done together in `encode_sample` specifically so one can't be forgotten; mirror logic lives in `encoding/moves.py::mirror_move_index`.
2. **MCTS value sign must flip at every backup level** (negamax convention: value is from the side-to-move's perspective). Getting it wrong makes the engine play against itself, but only looks "a bit weak" — no crash. Verify with a color-symmetry test: `eval(pos) == -eval(mirror(pos))`.
3. **Split by game-id, never by position.** Consecutive positions in a game are near-duplicates; a position-level split leaks the same game into train and val, producing falsely good validation loss. Gate 3 checks 0 shared game-ids across splits.
4. **Queen promotions go in the 56 queen-direction planes, not the 9 underpromotion planes** of the 8×8×73 move encoding. Mixing them breaks the encoding's bijectivity.
5. **Value head is the known weak point** since distillation from Stockfish is deliberately excluded. Phase 4.4 diagnostic: 200-game match, policy-only vs. policy+MCTS-400-sims; <150 Elo gain means the value head needs graduated mitigation.
6. python-chess move generation (~20-50k moves/s) may bottleneck MCTS more than the network forward pass — profile before rewriting in Cython/Rust; don't optimize without measuring.
7. Self-play draw rate climbing >70% signals insufficient opening diversity — mitigated via opening-book starts + temperature=1.0 for the first 15 plies.
8. MCTS network evaluations must batch **across parallel games** (64-128), not just within one tree, for real GPU utilization — required before the first RL iteration.
9. RL needs a hard stop criterion, set in `configs/rl.yaml` before running: abort if gating never passes within 10 iterations, abort if cumulative gain <50 Elo after 25 iterations, hard cap 40 iterations.
10. Alpha-beta pruned scores are bounds, not exact values — never compare/order moves by a pruned branch's returned score, only use it to pick the max (real bug, documented in `docs/DECISIONS.md` 2026-08-08).
11. Every reported Elo number needs a confidence interval — with 200 games the measurement error is ~±35 Elo; a 20-point "improvement" is noise. Repeated everywhere in project docs as "Rule #1."
12. Legal-move masking (illegal moves → -inf before softmax) must be identical between training and inference.
13. New training runs must pass an overfit test first: ~100% accuracy on 512 fixed positions before real training starts — failure means a bug, not a capacity limit.
14. Golden files/tensors (`tests/golden/`) must be bit-identical across Windows and macOS — treated as the encoder's platform-independence test.
15. Lichess PGN `Event` field is unreliable for time-control filtering pre-2017 — use the `TimeControl` field (base+increment seconds) instead.

## Custom subagent

`.claude/agents/chess-engine-expert.md` is a project-specific subagent with a symptom → root-cause diagnostic table for exactly these invariants (e.g. "validation accuracy well below 45%" → move-encoding index-flip; "draw rate >70% in self-play" → insufficient opening diversity). Prefer delegating chess-engine-correctness questions to it over reasoning from scratch.
