import 'package:chess_interface_dart/logical_interface/interface.dart';
import 'package:chess_interface_dart/logical_interface/piece.dart';
import 'package:flutter/foundation.dart';

import '../engine/engine.dart';
import '../engine/uci.dart';

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
/// [playerMove], [restart]). No game logic is scattered across widgets: what
/// happens can be understood from one file, and it can be exercised without
/// opening a window (see `test/game_controller_test.dart`).
class GameController extends ChangeNotifier {
  GameController(this._engine, {this.onBoardChanged});

  final Engine _engine;

  /// Called when the board changes because of the engine.
  ///
  /// `ChessBoardWidget` does not watch `GameController`: it repaints only when
  /// its own `ChessBoardProvider` tells it to. It handles the user's moves
  /// itself, but not the engine's — without this signal the piece moves in the
  /// data and stays put on screen.
  ///
  /// A plain function rather than the provider itself, so this file stays free
  /// of Flutter and remains drivable from tests.
  final void Function()? onBoardChanged;

  /// The position. `null` until a game is started.
  ChessBoardInterface? _game;
  ChessBoardInterface? get game => _game;

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
  /// This is the raw material for the end-of-game chart. It fills itself as
  /// you play and costs no extra request: the engine sends it alongside the
  /// move.
  final List<double> _evaluations = [];
  List<double> get evaluations => List.unmodifiable(_evaluations);

  /// The user always plays White: it is the simplest thing to explain, and it
  /// means they move first and never wait at the start.
  static const playerColor = PieceColor.white;

  void setDifficulty(Difficulty value) {
    if (_phase != GamePhase.menu) return;
    _difficulty = value;
    notifyListeners();
  }

  /// Starts a new game.
  void start() {
    _game = ChessBoardInterface();
    _phase = GamePhase.playing;
    _result = null;
    _resultDetail = null;
    _error = null;
    _evaluations.clear();
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
    notifyListeners();
  }

  /// Records that the user has moved, then lets the engine reply.
  ///
  /// The move has **already been applied** to the board by the widget, which
  /// validates it on its own: here we simply take note and carry on.
  Future<void> playerMove() async {
    final game = _game;
    if (game == null || _phase != GamePhase.playing) return;

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
      final move = await _engine.bestMove(game.toFEN(), _difficulty);
      _applyEngineMove(move);
      _evaluations.add(move.evaluation);
      onBoardChanged?.call();
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

  /// Applies the engine's chosen move to the board.
  ///
  /// Promotion has to be done separately: `move()` advances the pawn but does
  /// not transform it, and the engine may ask for a piece other than a queen.
  void _applyEngineMove(EngineMove move) {
    final game = _game;
    if (game == null) return;

    final from = uciFrom(move.uci);
    final to = uciTo(move.uci);

    if (!game.move(from, to)) {
      _error = 'The engine returned an illegal move (${move.uci}).';
      return;
    }

    final promotion = uciPromotion(move.uci);
    if (promotion != null) {
      game.promotePawn(to, promotion);
    }
  }

  /// If the game is over, updates the state and returns true.
  bool _checkGameOver() {
    final game = _game;
    if (game == null) return false;

    // Order matters: checkmate must be checked before stalemate, because in
    // both cases the player has no legal moves.
    if (game.isCheckmate()) {
      // The side to move is the one being mated: if it is the user, they lost.
      _finish(
        game.turn == playerColor ? GameResult.youLose : GameResult.youWin,
        'Checkmate',
      );
      return true;
    }

    if (game.isStalemate()) {
      _finish(GameResult.draw, 'Stalemate: no legal move, but no check either');
      return true;
    }

    if (game.isInsufficientMaterial()) {
      _finish(GameResult.draw, 'Neither side has enough material to mate');
      return true;
    }

    if (game.isThreefoldRepetition()) {
      _finish(GameResult.draw, 'Same position repeated three times');
      return true;
    }

    if (game.isFiftyMoveDraw()) {
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
