/// A game playing out, without opening a window.
///
/// The controller does not depend on Flutter, so it can be driven from a test.
/// That makes the endings (mate, stalemate, draw) verifiable — by hand they
/// would each require playing a whole game.
library;

import 'package:chess_interface_dart/logical_interface/interface.dart';
import 'package:chess_interface_dart/logical_interface/position.dart';
import 'package:chessbot_ui/engine/engine.dart';
import 'package:chessbot_ui/engine/uci.dart';
import 'package:chessbot_ui/game/game_controller.dart';
import 'package:flutter_test/flutter_test.dart';

/// An engine that plays moves dictated by the test.
class ScriptedEngine implements Engine {
  ScriptedEngine(this.moves, {this.evaluation = 0.0});

  final List<String> moves;
  final double evaluation;
  int _index = 0;

  @override
  Future<bool> ping() async => true;

  @override
  Future<EngineMove> bestMove(String fen, Difficulty level) async {
    if (_index >= moves.length) {
      throw const EngineException('The script ran out.');
    }
    return EngineMove(uci: moves[_index++], evaluation: evaluation);
  }
}

/// An engine that always fails, to exercise error handling.
class BrokenEngine implements Engine {
  @override
  Future<bool> ping() async => false;

  @override
  Future<EngineMove> bestMove(String fen, Difficulty level) async {
    throw const EngineException('The engine is not responding.');
  }
}

void main() {
  group('phases', () {
    test('starts at the menu', () {
      final c = GameController(ScriptedEngine([]));
      expect(c.phase, GamePhase.menu);
      expect(c.game, isNull);
    });

    test('starting a game builds the initial position', () {
      final c = GameController(ScriptedEngine([]))..start();
      expect(c.phase, GamePhase.playing);
      expect(c.game, isNotNull);
      expect(c.game!.toFEN(), startsWith('rnbqkbnr/pppppppp'));
    });

    test('restarting returns to the menu and clears everything', () {
      final c = GameController(ScriptedEngine([]))..start();
      c.restart();
      expect(c.phase, GamePhase.menu);
      expect(c.game, isNull);
      expect(c.evaluations, isEmpty);
    });

    test('difficulty can only be changed from the menu', () {
      final c = GameController(ScriptedEngine([]));
      c.setDifficulty(Difficulty.hard);
      expect(c.difficulty, Difficulty.hard);

      c.start();
      c.setDifficulty(Difficulty.easy);
      expect(c.difficulty, Difficulty.hard, reason: 'locked once the game has started');
    });
  });

  group('engine moves', () {
    test('the engine move lands on the board', () async {
      final c = GameController(ScriptedEngine(['e7e5']))..start();

      // White moves: the widget does this, here we do it by hand.
      c.game!.move(sq('e2'), sq('e4'));
      await c.playerMove();

      // After Black replies it is White to move again.
      expect(c.game!.toFEN(), contains(' w '));
      expect(c.engineThinking, isFalse);
    });

    test('the evaluation is recorded for the chart', () async {
      final c = GameController(ScriptedEngine(['e7e5'], evaluation: 0.3))
        ..start();

      c.game!.move(sq('e2'), sq('e4'));
      await c.playerMove();

      expect(c.evaluations, [0.3]);
    });

    test('a broken engine yields a readable error, not a crash', () async {
      final c = GameController(BrokenEngine())..start();

      c.game!.move(sq('e2'), sq('e4'));
      await c.playerMove();

      expect(c.error, isNotNull);
      expect(c.phase, GamePhase.playing, reason: 'the game does not end');
      expect(c.engineThinking, isFalse);
    });
  });

  group('endings', () {
    test("fool's mate makes the user lose", () async {
      // 1. f3 e5  2. g4 Qh4#  — Black (the engine) delivers mate.
      final c = GameController(ScriptedEngine(['e7e5', 'd8h4']))..start();

      c.game!.move(sq('f2'), sq('f3'));
      await c.playerMove();

      c.game!.move(sq('g2'), sq('g4'));
      await c.playerMove();

      expect(c.phase, GamePhase.over);
      expect(c.result, GameResult.youLose);
      expect(c.resultDetail, 'Checkmate');
    });
  });
}

/// Shorthand that keeps the tests readable: `sq('e2')` instead of
/// `Position(row: 6, col: 4)`.
Position sq(String square) => squareToPosition(square);
