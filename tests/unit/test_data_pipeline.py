"""Gate 3 — filtri, parsing PGN e storage.

I PGN di prova sono scritti a mano nel file: piccoli, leggibili, e ognuno copre un caso
che i filtri devono trattare in modo diverso. Un test che dipende da un dump scaricato
non e un test che gira ad ogni commit.
"""

from __future__ import annotations

import random
from pathlib import Path

import chess
import numpy as np
import pytest

from chessbot.data.filters import (
    FilterConfig,
    FilterStats,
    accept_headers,
    average_elo,
    category,
    estimated_duration,
    is_timeout_in_non_lost_position,
)
from chessbot.data.pgn import WDL_DRAW, WDL_LOSS, WDL_WIN, Sample, iter_samples
from chessbot.data.storage import (
    NO_EVAL,
    SAMPLE_DTYPE,
    bytes_per_sample,
    encode_samples,
    iter_shard_boards,
    pack_board,
    read_shard,
    unpack_board,
    write_shard,
)

from .conftest_positions import random_positions

pytestmark = pytest.mark.fast


PGN_BUONO = """[Event "Rated Classical game"]
[Site "https://lichess.org/abcd1234"]
[White "alice"]
[Black "bob"]
[Result "1-0"]
[WhiteElo "2200"]
[BlackElo "2150"]
[TimeControl "600+5"]
[Termination "Normal"]

1. e4 { [%eval 0.24] } e5 { [%eval 0.18] } 2. Nf3 { [%eval 0.31] } Nc6 { [%eval 0.25] }
3. Bb5 { [%eval 0.28] } a6 { [%eval 0.33] } 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6
8. c3 O-O 9. h3 Nb8 10. d4 Nbd7 11. Nbd2 Bb7 12. Bc2 Re8 1-0

"""

PGN_BULLET = """[Event "Rated Bullet game"]
[Site "https://lichess.org/bullet01"]
[Result "0-1"]
[WhiteElo "2400"]
[BlackElo "2450"]
[TimeControl "60+0"]
[Termination "Normal"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O
9. h3 Nb8 10. d4 Nbd7 0-1

"""

PGN_ELO_BASSO = """[Event "Rated Classical game"]
[Site "https://lichess.org/lowelo01"]
[Result "1-0"]
[WhiteElo "1400"]
[BlackElo "1350"]
[TimeControl "600+5"]
[Termination "Normal"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O
9. h3 Nb8 10. d4 Nbd7 1-0

"""

PGN_CORTA = """[Event "Rated Classical game"]
[Site "https://lichess.org/short001"]
[Result "0-1"]
[WhiteElo "2300"]
[BlackElo "2300"]
[TimeControl "900+10"]
[Termination "Normal"]

1. e4 e5 2. Qh5 Nc6 0-1

"""

PGN_TIMEOUT_PATTA = """[Event "Rated Classical game"]
[Site "https://lichess.org/timeout01"]
[Result "1/2-1/2"]
[WhiteElo "2300"]
[BlackElo "2300"]
[TimeControl "600+5"]
[Termination "Time forfeit"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O
9. h3 Nb8 10. d4 Nbd7 1/2-1/2

"""


@pytest.fixture
def pgn_file(tmp_path: Path) -> Path:
    path = tmp_path / "campione.pgn"
    path.write_text(
        PGN_BUONO + PGN_BULLET + PGN_ELO_BASSO + PGN_CORTA + PGN_TIMEOUT_PATTA,
        encoding="utf-8",
    )
    return path


# --------------------------------------------------------------------------------------
# Filtri
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tc,attesa",
    [
        ("60+0", "bullet"),
        ("120+1", "bullet"),
        ("180+0", "blitz"),
        ("300+0", "blitz"),
        ("600+0", "rapid"),
        ("300+5", "rapid"),
        ("1800+0", "classical"),
        ("600+30", "classical"),
    ],
)
def test_categoria_dal_time_control(tc: str, attesa: str):
    """Le soglie Lichess su base + 40*incremento."""
    assert category(tc) == attesa


@pytest.mark.parametrize("tc", ["-", "", None, "abc", "600", "600+"])
def test_time_control_non_parsabile(tc):
    assert estimated_duration(tc) is None
    assert category(tc) is None


def test_media_elo():
    assert average_elo("2000", "2100") == 2050
    assert average_elo(2000, 2100) == 2050
    assert average_elo(None, "2100") is None
    assert average_elo("boh", "2100") is None


def test_timeout_in_posizione_non_persa():
    assert is_timeout_in_non_lost_position("Time forfeit", "1/2-1/2")
    assert not is_timeout_in_non_lost_position("Time forfeit", "1-0")
    assert not is_timeout_in_non_lost_position("Normal", "1/2-1/2")
    assert not is_timeout_in_non_lost_position(None, "1/2-1/2")


def test_accept_headers_accetta_la_partita_buona():
    headers = {
        "TimeControl": "600+5",
        "WhiteElo": "2200",
        "BlackElo": "2150",
        "Result": "1-0",
        "Termination": "Normal",
    }
    assert accept_headers(headers, FilterConfig(), FilterStats())


@pytest.mark.parametrize(
    "override,motivo",
    [
        ({"TimeControl": "60+0"}, "bullet"),
        ({"WhiteElo": "1200", "BlackElo": "1100"}, "Elo basso"),
        ({"Result": "*"}, "esito assente"),
        ({"TimeControl": "-"}, "time control non parsabile"),
        ({"Termination": "Time forfeit", "Result": "1/2-1/2"}, "timeout"),
    ],
)
def test_accept_headers_scarta(override: dict, motivo: str):
    headers = {
        "TimeControl": "600+5",
        "WhiteElo": "2200",
        "BlackElo": "2150",
        "Result": "1-0",
        "Termination": "Normal",
    }
    headers.update(override)
    assert not accept_headers(headers, FilterConfig(), FilterStats()), motivo


def test_le_statistiche_dei_filtri_spiegano_gli_scarti():
    stats = FilterStats()
    base = {
        "TimeControl": "600+5",
        "WhiteElo": "2200",
        "BlackElo": "2150",
        "Result": "1-0",
        "Termination": "Normal",
    }
    accept_headers(base, FilterConfig(), stats)
    accept_headers({**base, "TimeControl": "60+0"}, FilterConfig(), stats)
    accept_headers({**base, "WhiteElo": "1000", "BlackElo": "1000"}, FilterConfig(), stats)

    assert stats.total == 3
    assert stats.kept == 0  # kept lo incrementa iter_samples, non accept_headers
    assert sum(stats.rejected.values()) == 2
    assert "%" in stats.summary()


# --------------------------------------------------------------------------------------
# Parsing PGN
# --------------------------------------------------------------------------------------


def test_tiene_solo_la_partita_valida(pgn_file: Path):
    """Delle cinque partite del campione, una sola supera tutti i filtri."""
    stats = FilterStats()
    samples = list(iter_samples(pgn_file, FilterConfig(), stats=stats))

    assert stats.total == 5
    assert stats.kept == 1
    assert {s.game_id for s in samples} == {"abcd1234"}


def test_le_posizioni_sono_quelle_della_partita(pgn_file: Path):
    samples = list(iter_samples(pgn_file, FilterConfig()))
    # 12 mosse per parte = 24 semimosse; l'ultima posizione non genera campione.
    assert len(samples) == 24
    assert samples[0].fen == chess.STARTING_FEN
    assert samples[0].move_uci == "e2e4"
    assert samples[0].ply == 0
    assert samples[1].move_uci == "e7e5"


def test_ogni_mossa_e_legale_nella_sua_posizione(pgn_file: Path):
    """Verifica incrociata con python-chess: la coppia (FEN, mossa) deve essere coerente."""
    for sample in iter_samples(pgn_file, FilterConfig()):
        board = chess.Board(sample.fen)
        move = chess.Move.from_uci(sample.move_uci)
        assert move in board.legal_moves, f"{sample.move_uci} illegale in {sample.fen}"


def test_etichetta_wdl_dal_punto_di_vista_del_bianco(pgn_file: Path):
    samples = list(iter_samples(pgn_file, FilterConfig()))
    assert all(s.wdl == WDL_WIN for s in samples), "la partita di prova finisce 1-0"


def test_eval_stockfish_letto_dai_commenti(pgn_file: Path):
    samples = list(iter_samples(pgn_file, FilterConfig()))
    assert samples[0].eval_cp == 24, "[%eval 0.24] -> 24 centipedoni"
    assert samples[1].eval_cp == 18
    # Dalla settima semimossa in poi il PGN di prova non ha piu valutazioni.
    assert samples[10].eval_cp is None


def test_partita_troppo_corta_scartata(tmp_path: Path):
    path = tmp_path / "corta.pgn"
    path.write_text(PGN_CORTA, encoding="utf-8")
    assert list(iter_samples(path, FilterConfig())) == []


def test_max_games_limita(tmp_path: Path):
    path = tmp_path / "molte.pgn"
    path.write_text(PGN_BUONO * 5, encoding="utf-8")
    samples = list(iter_samples(path, FilterConfig(), max_games=2))
    assert len({s.game_id for s in samples}) == 1  # stesso Site nelle copie
    assert len(samples) == 48  # 2 partite x 24 posizioni


def test_wdl_costanti_distinte():
    assert len({WDL_LOSS, WDL_DRAW, WDL_WIN}) == 3


# --------------------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------------------


def test_dimensione_del_campione_come_da_piano():
    """§2.6: ~103 byte per posizione, ~2 GB per 20M."""
    assert bytes_per_sample() <= 112, f"{bytes_per_sample()} byte, troppo per §2.6"
    gb_per_20m = bytes_per_sample() * 20e6 / 1e9
    assert gb_per_20m < 2.5, f"{gb_per_20m:.1f} GB per 20M posizioni"


def test_round_trip_pack_unpack_bit_a_bit():
    """Criterio del Gate 3: scrivi, rileggi, i campi combaciano esattamente."""
    checked = 0
    for board in random_positions(n_games=25, seed=777):
        bitboards, state, clock = pack_board(board)
        rebuilt = unpack_board(bitboards, state, clock)

        assert rebuilt.piece_map() == board.piece_map()
        assert rebuilt.turn == board.turn
        assert rebuilt.castling_rights == board.castling_rights
        assert rebuilt.ep_square == board.ep_square
        assert rebuilt.halfmove_clock == min(board.halfmove_clock, 255)
        checked += 1
    assert checked >= 1_000


def test_shard_scritto_e_riletto_identico(tmp_path: Path, pgn_file: Path):
    samples = list(iter_samples(pgn_file, FilterConfig()))
    path = tmp_path / "shard.npy"

    written = write_shard(path, samples)
    assert written == len(samples)

    array = read_shard(path)
    assert array.dtype == SAMPLE_DTYPE
    assert len(array) == len(samples)

    # Round-trip completo: dallo shard si torna alle posizioni originali.
    for original, (board, move_index, wdl, eval_cp) in zip(
        samples, iter_shard_boards(array), strict=True
    ):
        atteso = chess.Board(original.fen)
        assert board.piece_map() == atteso.piece_map()
        assert board.turn == atteso.turn
        assert wdl == original.wdl
        assert eval_cp == original.eval_cp
        assert 0 <= move_index < 4672


def test_indice_mossa_nello_shard_e_orientato(pgn_file: Path):
    """Lo shard salva l'indice gia orientato al lato che muove (criticita #1).

    Verifica incrociata: si decodifica l'indice con la funzione inversa e si deve
    riottenere la mossa originale.
    """
    from chessbot.encoding import decode_move

    samples = list(iter_samples(pgn_file, FilterConfig()))
    array = encode_samples(samples)
    for sample, row in zip(samples, array, strict=True):
        board = chess.Board(sample.fen)
        assert decode_move(board, int(row["move_index"])) == chess.Move.from_uci(sample.move_uci)


def test_eval_assente_usa_il_sentinella():
    sample = Sample(game_id="g", fen=chess.STARTING_FEN, move_uci="e2e4", wdl=WDL_DRAW, ply=0)
    array = encode_samples([sample])
    assert array[0]["eval_cp"] == NO_EVAL
    _, _, _, eval_cp = next(iter_shard_boards(array))
    assert eval_cp is None


def test_eval_presente_sopravvive_al_round_trip():
    sample = Sample(
        game_id="g", fen=chess.STARTING_FEN, move_uci="e2e4", wdl=WDL_DRAW, ply=0, eval_cp=-137
    )
    array = encode_samples([sample])
    _, _, _, eval_cp = next(iter_shard_boards(array))
    assert eval_cp == -137


def test_shard_vuoto_non_esplode(tmp_path: Path):
    path = tmp_path / "vuoto.npy"
    assert write_shard(path, []) == 0
    assert len(read_shard(path)) == 0


def test_bitboard_coerenti_con_python_chess():
    """Verifica incrociata indipendente: i bitboard devono essere quelli di python-chess."""
    rng = random.Random(3)
    board = chess.Board()
    for _ in range(40):
        moves = list(board.legal_moves)
        if not moves:
            break
        board.push(rng.choice(moves))
        bitboards, _, _ = pack_board(board)
        assert int(bitboards[0]) == int(board.pieces(chess.PAWN, chess.WHITE))
        assert int(bitboards[11]) == int(board.pieces(chess.KING, chess.BLACK))
        assert np.count_nonzero([bin(int(b)).count("1") for b in bitboards]) <= 12
