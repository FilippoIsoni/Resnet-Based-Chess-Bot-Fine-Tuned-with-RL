"""Da PGN a posizioni (§2.2 del piano).

`python-chess` legge il PGN e rigioca le mosse una per una: ad ogni passo si ha in
memoria una `Board` corretta, da cui si estrae la fotografia logica dello stato.

Le posizioni sono campioni **indipendenti**: gli scacchi sono markoviani a informazione
perfetta, e cio che la posizione da sola non dice (arrocco, en passant, ripetizioni) e
gia codificato come piano esplicito dall'encoder. L'unico punto in cui la sequenza entra
e il *target del valore*, che viene dall'esito finale — ma entra come etichetta, non
come input.

I dump si leggono in **streaming** dal `.zst`: il 2026-07 sono 27 GB compressi e oltre
200 GB decompressi, che non stanno su disco e non servono.
"""

from __future__ import annotations

import io
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import chess
import chess.pgn

from .filters import FilterConfig, FilterStats, accept_headers

# Esito della partita dal punto di vista del BIANCO, come indice 0/1/2 per la testa WDL.
# L'ordine e fissato: loss, draw, win. Cambiarlo invalida i modelli allenati.
WDL_LOSS, WDL_DRAW, WDL_WIN = 0, 1, 2

_RESULT_TO_WDL = {"1-0": WDL_WIN, "0-1": WDL_LOSS, "1/2-1/2": WDL_DRAW}


@dataclass(slots=True)
class Sample:
    """Una posizione con le sue etichette, prima dell'encoding.

    Si tiene la FEN e non il tensore: espandere 19x8x8 float qui significherebbe 97 GB
    per 20M posizioni (§2.6). L'espansione avviene al volo nel DataLoader.
    """

    game_id: str
    """Identificatore della partita di provenienza. E la chiave dello split per PARTITA
    (criticita #3): senza, non si puo dimostrare l'assenza di leakage."""

    fen: str
    move_uci: str
    """Mossa giocata, in coordinate assolute. L'orientamento al lato che muove lo
    applica `encoding.encode_move`, che sa anche flippare l'indice (criticita #1)."""

    wdl: int
    """Esito finale dal punto di vista del BIANCO: 0 loss, 1 draw, 2 win."""

    ply: int
    eval_cp: int | None = None
    """Valutazione Stockfish in centipedoni dal punto di vista del bianco, se il PGN la
    riporta come `[%eval ...]`. Circa un terzo delle partite filtrate ce l'ha."""


def _parse_eval(comment: str) -> int | None:
    """Estrae `[%eval ...]` da un commento, in centipedoni dal lato del bianco.

    Il campo puo essere un numero in pedoni (`-1.24`) o un matto (`#-3`). I matti si
    convertono in un valore grande ma finito: servono come segnale di "vinta/persa", e
    un infinito romperebbe qualunque loss numerica a valle.
    """
    marker = "[%eval "
    start = comment.find(marker)
    if start < 0:
        return None
    start += len(marker)
    end = comment.find("]", start)
    if end < 0:
        return None
    token = comment[start:end].strip()

    if token.startswith("#"):
        try:
            mate_in = int(token[1:])
        except ValueError:
            return None
        # Matto piu vicino = valore piu estremo, ma sempre entro un limite gestibile.
        magnitude = max(10_000 - abs(mate_in) * 100, 5_000)
        return magnitude if mate_in > 0 else -magnitude

    try:
        return round(float(token) * 100)
    except ValueError:
        return None


def iter_games(path: Path) -> Iterator[chess.pgn.Game]:
    """Legge tutte le partite, senza filtrare — utile per ispezionare un PGN a mano.

    La pipeline usa `iter_samples`, che filtra sugli header prima di costruire l'albero
    ed e due ordini di grandezza piu veloce sui dump veri.
    """
    with _open_stream(path) as stream:
        while (game := chess.pgn.read_game(stream)) is not None:
            yield game


@contextmanager
def _open_stream(path: Path) -> Iterator[TextIO]:
    """Apre un `.pgn` o `.pgn.zst` come stream di testo, in streaming.

    Context manager perche il caso `.zst` incatena tre oggetti (file, decompressore,
    wrapper di testo): chiuderli a mano lascerebbe il file aperto se qualcosa fallisce
    a meta.
    """
    if path.suffix == ".zst":
        import zstandard

        with open(path, "rb") as raw:
            reader = zstandard.ZstdDecompressor().stream_reader(raw)
            yield io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
    else:
        with open(path, encoding="utf-8", errors="replace") as stream:
            yield stream


class _FilteringVisitor(chess.pgn.BaseVisitor):
    """Costruisce l'albero delle mosse SOLO per le partite che passano i filtri.

    > **Perche non basta `read_game`.** Su un dump Lichess i filtri scartano oltre il 99%
    > delle partite (misurato: 0.86% tenute sul 2026-07), ma `read_game` costruisce
    > comunque l'albero completo di ognuna. Misurato sul dump vero: 94 partite/s contro
    > le 1.702 di questo visitor. Un fattore 18, che sulla conversione e la differenza
    > fra 27 ore e un'ora e mezza.
    >
    > Il gancio e `end_headers`: restituendo `SKIP`, il parser salta il corpo senza
    > parsarlo. Non si puo ottenere lo stesso con `read_headers` + salto manuale, perche
    > `read_headers` consuma gia il corpo e si ferma sull'`[Event` successivo — lo stream
    > non e piu posizionato sulle mosse, e non c'e nulla da saltare.
    """

    def __init__(self, accept: Callable[[chess.pgn.Headers], bool]) -> None:
        self._accept = accept
        self.game = chess.pgn.Game()
        self.skipped = False
        self._stack: list[chess.pgn.GameNode] = [self.game]

    def begin_game(self) -> None:
        self.game = chess.pgn.Game()
        self.skipped = False
        self._stack = [self.game]

    def visit_header(self, tagname: str, tagvalue: str) -> None:
        self.game.headers[tagname] = tagvalue

    def end_headers(self) -> chess.pgn.SkipType | None:
        if not self._accept(self.game.headers):
            self.skipped = True
            return chess.pgn.SKIP
        return None

    def visit_move(self, board: chess.Board, move: chess.Move) -> None:
        self._stack[-1] = self._stack[-1].add_variation(move)

    def visit_comment(self, comment: str) -> None:
        self._stack[-1].comment = comment

    def handle_error(self, error: Exception) -> None:
        # Un PGN malformato non deve interrompere la conversione: la partita viene
        # troncata al punto in cui smette di avere senso, ed e `samples_from_game` a
        # decidere se cio che resta e abbastanza.
        pass

    def result(self) -> chess.pgn.Game:
        return self.game

    def as_factory(self) -> Callable[[], _FilteringVisitor]:
        """Adatta l'istanza all'API di `read_game`, che si aspetta una factory."""

        def factory() -> _FilteringVisitor:
            return self

        return factory


def game_id_of(game: chess.pgn.Game, fallback: int) -> str:
    """Identificatore stabile della partita.

    Lichess mette l'URL in `Site`, che finisce con l'id della partita: e la chiave
    naturale. Senza, si usa un progressivo — che e stabile solo a parita di file e
    ordine, quindi va evitato quando l'id vero esiste.
    """
    site = game.headers.get("Site", "")
    if "lichess.org/" in site:
        return site.rsplit("/", 1)[-1]
    return f"game{fallback:09d}"


def samples_from_game(game: chess.pgn.Game, game_id: str, *, min_moves: int) -> list[Sample]:
    """Rigioca la partita e produce un campione per posizione.

    L'ultima posizione non genera campione: non c'e una mossa da predire.
    """
    result = game.headers.get("Result", "")
    wdl = _RESULT_TO_WDL.get(result)
    if wdl is None:
        return []

    board = game.board()
    samples: list[Sample] = []

    for ply, node in enumerate(game.mainline()):
        move = node.move
        if move is None or move not in board.legal_moves:
            # Un PGN malformato non deve far cadere l'intera conversione: si scarta la
            # partita dal punto in cui smette di avere senso.
            break
        samples.append(
            Sample(
                game_id=game_id,
                fen=board.fen(),
                move_uci=move.uci(),
                wdl=wdl,
                ply=ply,
                eval_cp=_parse_eval(node.comment) if node.comment else None,
            )
        )
        board.push(move)

    # `min_moves` e in mosse intere, i campioni sono in semimosse.
    if len(samples) < min_moves * 2:
        return []
    return samples


def iter_samples(
    path: Path,
    config: FilterConfig | None = None,
    *,
    stats: FilterStats | None = None,
    max_games: int | None = None,
) -> Iterator[Sample]:
    """Pipeline completa: PGN -> partite filtrate -> posizioni.

    I filtri sugli header decidono se costruire o no l'albero delle mosse: vedi
    `_FilteringVisitor`, che e il motivo per cui la conversione dura un'ora e mezza
    invece di 27.
    """
    config = config or FilterConfig()
    stats = stats if stats is not None else FilterStats()

    def accept(headers: chess.pgn.Headers) -> bool:
        return accept_headers(dict(headers), config, stats)

    accepted = 0
    index = 0
    with _open_stream(path) as stream:
        while True:
            # `read_game` vuole una FACTORY di visitor, non un'istanza: ne creerebbe uno
            # nuovo ad ogni partita. Qui serve invece tenere il riferimento per leggere
            # `skipped` dopo, quindi la factory restituisce sempre lo stesso oggetto —
            # che va bene perche `begin_game` lo reinizializza.
            visitor = _FilteringVisitor(accept)
            game = chess.pgn.read_game(stream, Visitor=visitor.as_factory())
            if game is None:
                return

            index += 1
            if visitor.skipped:
                continue

            samples = samples_from_game(
                game, game_id_of(game, index - 1), min_moves=config.min_moves
            )
            if not samples:
                stats.reject(f"meno di {config.min_moves} mosse")
                continue

            stats.kept += 1
            yield from samples

            accepted += 1
            if max_games is not None and accepted >= max_games:
                return
