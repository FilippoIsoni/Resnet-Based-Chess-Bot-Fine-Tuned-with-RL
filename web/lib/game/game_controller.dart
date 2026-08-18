import 'package:chess/chess.dart' as ch;
import 'package:flutter/foundation.dart';

import '../engine/engine.dart';

/// Which phase the app is in.
enum GamePhase {
  /// Opening screen: pick a difficulty and press start.
  menu,

  /// A game is in progress.
  playing,

  /// The game is over: show the result and the chart.
  over,
}

/// How the game ended.
enum GameResult {
  youWin('You win'),
  youLose('You lose'),
  draw('Draw');

  const GameResult(this.label);
  final String label;
}

/// The game state, and the only place it is mutated.
///
/// The UI reads from here and calls the three public methods ([start],
/// [playMove], [restart]). No chess logic is scattered across widgets: what
/// happens can be understood from one file, and it can be exercised without
/// opening a window (see `test/game_controller_test.dart`).
///
/// The rules come from the `chess` package. It replaced `chess_interface`,
/// which allowed a king to be captured and never updated castling rights when
/// a rook moved — producing positions the backend rightly refused. Everything
/// here trusts the library for legality and never second-guesses it.
class GameController extends ChangeNotifier {
  GameController(this._engine);

  final Engine _engine;

  /// The position. `null` until a game is started.
  ch.Chess? _game;
  ch.Chess? get game => _game;

  GamePhase _phase = GamePhase.menu;
  GamePhase get phase => _phase;

  Difficulty _difficulty = Difficulty.medium;
  Difficulty get difficulty => _difficulty;

  GameResult? _result;
  GameResult? get result => _result;

  /// Why the game ended: "Checkmate", "Stalemate", ...
  String? _resultDetail;
  String? get resultDetail => _resultDetail;

  bool _engineThinking = false;
  bool get engineThinking => _engineThinking;

  /// Error to display, if the engine's last move failed.
  String? _error;
  String? get error => _error;

  /// The evaluation after each engine move, from White's point of view.
  ///
  /// The raw material for the end-of-game chart. It fills itself as you play
  /// and costs no extra request: the engine sends it alongside the move.
  final List<double> _evaluations = [];
  List<double> get evaluations => List.unmodifiable(_evaluations);

  /// The move just played, in UCI, for the board to highlight.
  String? _lastMove;
  String? get lastMove => _lastMove;

  /// The moves played so far, in SAN, for the history list.
  final List<String> _history = [];
  List<String> get history => List.unmodifiable(_history);

  /// The user always plays White: it is the simplest thing to explain, and it
  /// means they move first and never wait at the start.
  static const playerIsWhite = true;

  void setDifficulty(Difficulty value) {
    if (_phase != GamePhase.menu) return;
    _difficulty = value;
    notifyListeners();
  }

  /// Starts a new game.
  void start() {
    _game = ch.Chess();
    _phase = GamePhase.playing;
    _result = null;
    _resultDetail = null;
    _error = null;
    _evaluations.clear();
    _history.clear();
    _lastMove = null;
    notifyListeners();
  }

  /// Returns to the opening menu.
  void restart() {
    _game = null;
    _phase = GamePhase.menu;
    _result = null;
    _resultDetail = null;
    _error = null;
    _engineThinking = false;
    _evaluations.clear();
    _history.clear();
    _lastMove = null;
    notifyListeners();
  }

  /// Applies the player's move, then lets the engine reply.
  ///
  /// [uci] comes from the board widget, which only offers legal destinations —
  /// but it is applied through the library all the same, so an illegal move
  /// could never slip through even if the widget had a bug.
  Future<void> playMove(String uci) async {
    final game = _game;
    if (game == null || _phase != GamePhase.playing || _engineThinking) return;

    if (!_apply(game, uci)) {
      _error = 'That move is not legal.';
      notifyListeners();
      return;
    }

    _error = null;
    notifyListeners();

    if (_checkGameOver()) return;
    await _engineMove();
  }

  /// Asks the engine for a move and applies it.
  Future<void> _engineMove() async {
    final game = _game;
    if (game == null) return;

    _engineThinking = true;
    _error = null;
    notifyListeners();

    try {
      final move = await _engine.bestMove(game.fen, _difficulty);
      if (!_apply(game, move.uci)) {
        _error =
            'The engine returned a move this position does not allow '
            '(${move.uci}).';
      } else {
        _evaluations.add(move.evaluation);
      }
    } on EngineException catch (e) {
      _error = e.message;
    } catch (_) {
      _error = 'Unexpected error while contacting the engine.';
    } finally {
      _engineThinking = false;
      notifyListeners();
    }

    _checkGameOver();
  }

  /// Applies one UCI move, recording its SAN for the history.
  ///
  /// Returns false if the library rejects it — which should not happen, and is
  /// surfaced rather than swallowed if it does.
  bool _apply(ch.Chess game, String uci) {
    final request = {
      'from': uci.substring(0, 2),
      'to': uci.substring(2, 4),
      if (uci.length > 4) 'promotion': uci[4],
    };

    // SAN has to be read before the move is made: it describes the move in the
    // position it is played from.
    final san = _sanFor(game, request);
    if (!game.move(request)) return false;

    _history.add(san ?? uci);
    _lastMove = uci;
    return true;
  }

  String? _sanFor(ch.Chess game, Map<String, String> request) {
    for (final m in game.generate_moves()) {
      if (ch.Chess.algebraic(m.from) == request['from'] &&
          ch.Chess.algebraic(m.to) == request['to']) {
        return game.move_to_san(m);
      }
    }
    return null;
  }

  /// If the game is over, updates the state and returns true.
  bool _checkGameOver() {
    final game = _game;
    if (game == null) return false;

    // Checkmate before stalemate: in both the side to move has no legal
    // moves, and only the check tells them apart.
    if (game.in_checkmate) {
      // The side to move is the one being mated.
      final userIsMated = (game.turn == ch.Color.WHITE) == playerIsWhite;
      _finish(
        userIsMated ? GameResult.youLose : GameResult.youWin,
        'Checkmate',
      );
      return true;
    }

    if (game.in_stalemate) {
      _finish(GameResult.draw, 'Stalemate: no legal move, but no check either');
      return true;
    }

    if (game.insufficient_material) {
      _finish(GameResult.draw, 'Neither side has enough material to mate');
      return true;
    }

    if (game.in_threefold_repetition) {
      _finish(GameResult.draw, 'Same position repeated three times');
      return true;
    }

    if (game.half_moves >= 100) {
      _finish(GameResult.draw, 'Fifty moves with no capture and no pawn move');
      return true;
    }

    return false;
  }

  void _finish(GameResult result, String detail) {
    _result = result;
    _resultDetail = detail;
    _phase = GamePhase.over;
    notifyListeners();
  }
}
