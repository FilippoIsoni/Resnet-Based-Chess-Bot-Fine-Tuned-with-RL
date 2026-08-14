"""Expert Iteration — il ciclo dell'RL (Stadio 6).

    python scripts/run_rl.py                        parte o RIPRENDE automaticamente
    python scripts/run_rl.py --iterations 25
    python scripts/run_rl.py --run prova --games 20 --sims 40    smoke test
    python scripts/run_rl.py --fresh                ricomincia da zero

Ogni iterazione:

    1. SELF-PLAY   N partite in parallelo, MCTS con rumore di Dirichlet
    2. TRAINING    riallena sulla distribuzione delle visite (replay buffer)
    3. GATING      la rete nuova sfida la corrente: promossa solo se vince

## La ripresa e automatica

Come per il training supervisionato: se `runs/rl/<nome>/state.pt` esiste, si riprende da
li — pesi correnti, replay buffer, storico delle iterazioni. Dopo un crash o uno standby
basta rilanciare lo stesso comando.

## I criteri di stop sono in configs/rl.yaml

Non "a sentimento" (criticita #9). Il ciclo si ferma da solo se il gating non passa mai
in 10 iterazioni, se il guadagno cumulativo e sotto 50 Elo dopo 25, o al tetto di 40.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import chess  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from chessbot.model.network import ChessNet, NetworkConfig  # noqa: E402
from chessbot.search import Evaluator, SearchConfig, run_mcts  # noqa: E402
from chessbot.search.openings import (  # noqa: E402
    build_book_from_dataset,
    load_book,
    sample_openings,
    save_book,
)
from chessbot.search.parallel import self_play_batch  # noqa: E402
from chessbot.training.checkpoint import (  # noqa: E402
    CheckpointMeta,
    HistoryLog,
    commit_hash,
    load_checkpoint,
    save_checkpoint,
)
from chessbot.training.expert_iteration import (  # noqa: E402
    IterationResult,
    ReplayBuffer,
    build_rl_optimizer,
    load_rl_config,
    records_to_tensors,
    rl_loss,
    should_stop,
)
from chessbot.training.gating import SprtTest, SprtVerdict  # noqa: E402
from chessbot.utils.seed import set_seed  # noqa: E402

_COLOR = sys.stdout.isatty()
GREEN, RED, YELLOW, GREY, BOLD, RESET = (
    ("\033[32m", "\033[31m", "\033[33m", "\033[90m", "\033[1m", "\033[0m")
    if _COLOR
    else ("", "", "", "", "", "")
)


def ensure_book(path: Path, data_dir: Path) -> list[chess.Board]:
    """Carica il libro d'aperture, generandolo dal dataset se non c'e."""
    if not path.exists():
        print("  libro assente, lo genero dal dataset...")
        fens = build_book_from_dataset(data_dir, shards_to_scan=4, max_positions=400)
        save_book(path, fens, source="generato da run_rl.py")
        print(f"  {len(fens)} posizioni di apertura salvate in {path.relative_to(ROOT)}")
    return load_book(path)


def train_on_buffer(model, optimizer, buffer, reference_model, config, device) -> dict[str, float]:
    """Riallena la rete sul replay buffer. Restituisce le loss medie."""
    records = buffer.all_records()
    if not records:
        return {}

    planes, policy_t, value_t, masks = records_to_tensors(records, config)
    n = len(records)
    model.train()

    totals: dict[str, float] = {}
    steps = 0

    for _epoch in range(config.epochs_per_iteration):
        perm = torch.randperm(n)
        for start in range(0, n, config.batch_size):
            idx = perm[start : start + config.batch_size]
            if len(idx) < 2:
                continue

            b_planes = planes[idx].to(device)
            b_policy = policy_t[idx].to(device)
            b_value = value_t[idx].to(device)
            b_mask = masks[idx].to(device)

            policy_logits, wdl_logits = model(b_planes)

            reference_logits = None
            if reference_model is not None:
                with torch.no_grad():
                    reference_logits, _ = reference_model(b_planes)

            loss, parts = rl_loss(
                policy_logits, wdl_logits, b_policy, b_value, b_mask, reference_logits, config
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            for k, v in parts.items():
                totals[k] = totals.get(k, 0.0) + v
            steps += 1

    model.eval()
    return {k: v / max(steps, 1) for k, v in totals.items()}


def play_gating_match(
    challenger, champion, openings, config, device, rng, *, max_games: int
) -> SprtTest:
    """La rete nuova sfida quella corrente. SPRT: si ferma appena sa abbastanza."""
    test = SprtTest(max_games=max_games)
    search = SearchConfig(simulations=config.simulations_per_move)

    for i in range(max_games):
        board = sample_openings(openings, 1, rng)[0]
        challenger_white = i % 2 == 0

        while not board.is_game_over(claim_draw=True) and board.ply() < config.max_plies:
            is_challenger = (board.turn == chess.WHITE) == challenger_white
            evaluator = challenger if is_challenger else champion
            result = run_mcts(board, evaluator, search, rng=rng)
            if result.move is None:
                break
            board.push(result.move)

        outcome = (
            board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else "1/2-1/2"
        )
        if outcome == "1/2-1/2":
            score = 0.5
        elif (outcome == "1-0") == challenger_white:
            score = 1.0
        else:
            score = 0.0

        if test.update(score) != SprtVerdict.CONTINUE:
            break

    return test


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="rl", help="nome della cartella in runs/rl/")
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "runs" / "supervised" / "best.pt")
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "processed")
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "rl.yaml")
    ap.add_argument("--iterations", type=int, default=None, help="default: dal config")
    ap.add_argument("--games", type=int, default=None, help="partite per iterazione")
    ap.add_argument("--sims", type=int, default=None, help="simulazioni per mossa")
    ap.add_argument("--gating-games", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = ROOT / "runs" / "rl" / args.run
    run_dir.mkdir(parents=True, exist_ok=True)

    config = load_rl_config(args.config)
    if args.games:
        config.games_per_iteration = args.games
    if args.sims:
        config.simulations_per_move = args.sims
    if args.iterations:
        config.max_iterations = args.iterations

    print(f"{BOLD}Expert Iteration — Stadio 6{RESET}")
    print(f"  run: {run_dir}")
    print(
        f"  {config.games_per_iteration} partite/iter x {config.simulations_per_move} sim, "
        f"{config.parallel_games} in parallelo"
    )
    print(f"  gating: SPRT fino a {args.gating_games} partite, soglia 55%")
    print(
        f"  stop: {config.max_iterations} iter max, "
        f"gating entro {config.abort_if_no_gating_pass_within}, "
        f"{config.abort_if_cumulative_elo_below:.0f} Elo dopo {config.abort_elo_check_at}"
    )
    print()

    openings = ensure_book(ROOT / "data" / "books" / "openings.json", args.data)
    print(f"  libro: {len(openings)} posizioni di apertura")

    # --- reti -------------------------------------------------------------------------
    champion = ChessNet(NetworkConfig()).to(device)
    load_checkpoint(args.checkpoint, model=champion, map_location=device)
    champion.eval()

    # La rete supervisionata resta fissa come ancora per la penalita KL (§5.2).
    reference = ChessNet(NetworkConfig()).to(device)
    load_checkpoint(args.checkpoint, model=reference, map_location=device)
    reference.eval()
    for p in reference.parameters():
        p.requires_grad_(False)

    buffer = ReplayBuffer(max_iterations=config.replay_buffer_iterations)
    results: list[IterationResult] = []
    start_iter = 0

    # --- ripresa ----------------------------------------------------------------------
    state_path = run_dir / "state.pt"
    if state_path.exists() and not args.fresh:
        payload = torch.load(state_path, map_location=device, weights_only=False)
        champion.load_state_dict(payload["champion"])
        start_iter = payload["iteration"]
        results = [IterationResult(**r) for r in payload["results"]]
        print(f"  {BOLD}ripresa dall'iterazione {start_iter}{RESET}")
        promossi = sum(1 for r in results if r.gating_accepted)
        print(f"  storico: {len(results)} iterazioni, {promossi} promozioni")

    history = HistoryLog(run_dir / "history.jsonl")
    rng = np.random.default_rng(args.seed + start_iter)
    print()

    for iteration in range(start_iter, config.max_iterations):
        stop, reason = should_stop(results, config)
        if stop:
            print(f"{YELLOW}STOP: {reason}{RESET}")
            break

        print(f"{BOLD}--- iterazione {iteration + 1}/{config.max_iterations} ---{RESET}")
        started = time.perf_counter()

        # --- 1. SELF-PLAY -------------------------------------------------------------
        champion_eval = Evaluator(champion, device)
        search = SearchConfig(
            simulations=config.simulations_per_move,
            dirichlet_alpha=config.dirichlet_alpha,
            dirichlet_epsilon=config.dirichlet_weight,
        )

        records = []
        stats_totali = {"games": 0, "positions": 0, "draws": 0}
        rimanenti = config.games_per_iteration
        while rimanenti > 0:
            lotto = min(config.parallel_games, rimanenti)
            recs, st = self_play_batch(
                champion_eval,
                n_games=lotto,
                config=search,
                openings=sample_openings(openings, lotto, rng),
                max_plies=config.max_plies,
                temperature_moves=config.temperature_moves,
                rng=rng,
            )
            records.extend(recs)
            stats_totali["games"] += st.games
            stats_totali["positions"] += st.positions
            stats_totali["draws"] += st.results.get("1/2-1/2", 0)
            rimanenti -= lotto
            print(
                f"    self-play {stats_totali['games']}/{config.games_per_iteration} "
                f"({stats_totali['positions']:,} posizioni, {st.games_per_hour:.0f} partite/h)",
                flush=True,
            )

        draw_rate = stats_totali["draws"] / max(stats_totali["games"], 1)
        if draw_rate > config.max_draw_rate:
            print(
                f"    {YELLOW}patte {100 * draw_rate:.0f}% sopra la soglia "
                f"{100 * config.max_draw_rate:.0f}%: serve piu diversita nelle aperture{RESET}"
            )

        buffer.add(records)

        # --- 2. TRAINING --------------------------------------------------------------
        challenger = ChessNet(NetworkConfig()).to(device)
        challenger.load_state_dict(champion.state_dict())
        optimizer = build_rl_optimizer(challenger, config)

        losses = train_on_buffer(challenger, optimizer, buffer, reference, config, device)
        print(
            f"    training su {len(buffer):,} posizioni: "
            f"policy {losses.get('policy', 0):.4f}, value {losses.get('value', 0):.4f}, "
            f"kl {losses.get('kl', 0):.4f}"
        )

        # --- 3. GATING ----------------------------------------------------------------
        test = play_gating_match(
            Evaluator(challenger, device),
            Evaluator(champion, device),
            openings,
            config,
            device,
            rng,
            max_games=args.gating_games,
        )
        accepted = test.verdict() == SprtVerdict.ACCEPT
        elo, _margin = test.elo_estimate()
        elo = 0.0 if not np.isfinite(elo) else elo

        colore = GREEN if accepted else RED
        print(
            f"    gating: {colore}{'PROMOSSA' if accepted else 'scartata'}{RESET} — {test.summary()}"
        )

        if accepted:
            champion.load_state_dict(challenger.state_dict())
            save_checkpoint(
                run_dir / f"champion_iter{iteration + 1:03d}.pt",
                model=champion,
                meta=CheckpointMeta(notes=f"RL iter {iteration + 1}, {test.summary()}"),
            )

        elapsed = time.perf_counter() - started
        result = IterationResult(
            iteration=iteration + 1,
            games=stats_totali["games"],
            positions=stats_totali["positions"],
            draw_rate=draw_rate,
            gating_accepted=accepted,
            gating_summary=test.summary(),
            elo_estimate=elo,
            losses=losses,
            elapsed_seconds=round(elapsed, 1),
        )
        results.append(result)
        history.append({"kind": "rl_iteration", **asdict(result)})

        # --- checkpoint dell'iterazione ------------------------------------------------
        torch.save(
            {
                "champion": champion.state_dict(),
                "iteration": iteration + 1,
                "results": [asdict(r) for r in results],
                "commit": commit_hash(),
                "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
            run_dir / "state.pt.tmp",
        )
        (run_dir / "state.pt.tmp").replace(run_dir / "state.pt")

        cumulativo = sum(r.elo_estimate for r in results if r.gating_accepted)
        print(
            f"    {elapsed / 60:.1f} min | promozioni {sum(1 for r in results if r.gating_accepted)}"
            f"/{len(results)} | Elo cumulativo {cumulativo:+.0f}"
        )
        print()

    # --- riepilogo ----------------------------------------------------------------------
    print(f"{BOLD}Riepilogo{RESET}")
    promossi = [r for r in results if r.gating_accepted]
    cumulativo = sum(r.elo_estimate for r in promossi)
    print(f"  iterazioni: {len(results)}")
    print(f"  promozioni: {len(promossi)}")
    print(f"  Elo cumulativo stimato: {cumulativo:+.0f}")
    if results:
        print(f"  patte medie: {100 * sum(r.draw_rate for r in results) / len(results):.0f}%")
        print(f"  tempo totale: {sum(r.elapsed_seconds for r in results) / 3600:.1f} h")

    summary = {
        "timestamp_utc": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
        "commit": commit_hash(),
        "iterations": len(results),
        "promotions": len(promossi),
        "cumulative_elo": cumulativo,
        "results": [asdict(r) for r in results],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"  salvato in {(run_dir / 'summary.json').relative_to(ROOT)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
