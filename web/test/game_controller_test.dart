/// A game playing out, without opening a window.
///
/// The controller does not depend on Flutter widgets, so it can be driven from
/// a test. That makes the endings (mate, stalemate, draw) verifiable — by hand
/// each would need a whole game played by mouse.
///
/// **The FEN legality tests are the ones that matter.** The previous chess
/// library produced positions chess does not allow — it never updated castling
/// rights when a rook moved — and the backend rejected them mid-game. The old
/// tests missed it because they only checked that moves were *accepted*, never
/// that the resulting position was legal. These check the position itself.
library;

import 'package:chess/chess.dart' as ch;
import 'package:chessbot_ui/engine/engine.dart';
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

/// An engine that replies with whatever the rules library would play first.
/// Useful for running long games without scripting every move.
class FirstLegalEngine implements Engine {
  @override
  Future<bool> ping() async => true;

  @override
  Future<EngineMove> bestMove(String fen, Difficulty level) async {
    final game = ch.Chess.fromFEN(fen);
    final moves = game.generate_moves();
    if (moves.isEmpty) throw const EngineException('No move.');
    final m = moves.first;
    final promo = m.promotion == null ? '' : m.promotion!.toLowerCase();
    return EngineMove(
      uci: '${ch.Chess.algebraic(m.from)}${ch.Chess.algebraic(m.to)}$promo',
      evaluation: 0.0,
    );
  }
}

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
      expect(c.game!.fen, startsWith('rnbqkbnr/pppppppp'));
    });

    test('restarting returns to the menu and clears everything', () {
      final c = GameController(ScriptedEngine([]))..start();
      c.restart();
      expect(c.phase, GamePhase.menu);
      expect(c.game, isNull);
      expect(c.evaluations, isEmpty);
      expect(c.history, isEmpty);
    });

    test('difficulty can only be changed from the menu', () {
      final c = GameController(ScriptedEngine([]));
      c.setDifficulty(Difficulty.hard);
      expect(c.difficulty, Difficulty.hard);

      c.start();
      c.setDifficulty(Difficulty.easy);
      expect(
        c.difficulty,
        Difficulty.hard,
        reason: 'locked once the game has started',
      );
    });
  });

  group('moves', () {
    test('the engine move lands on the board', () async {
      final c = GameController(ScriptedEngine(['e7e5']))..start();
      await c.playMove('e2e4');

      expect(
        c.game!.fen,
        contains(' w '),
        reason: 'back to White after Black replies',
      );
      expect(c.history, ['e4', 'e5']);
      expect(c.engineThinking, isFalse);
    });

    test('an illegal move from the player is refused', () async {
      final c = GameController(ScriptedEngine([]))..start();
      final before = c.game!.fen;
      await c.playMove('e2e5'); // pawns do not jump three squares

      expect(c.game!.fen, before, reason: 'position untouched');
      expect(c.error, isNotNull);
    });

    test('the evaluation is recorded for the chart', () async {
      final c = GameController(ScriptedEngine(['e7e5'], evaluation: 0.3))
        ..start();
      await c.playMove('e2e4');
      expect(c.evaluations, [0.3]);
    });

    test('a broken engine yields a readable error, not a crash', () async {
      final c = GameController(BrokenEngine())..start();
      await c.playMove('e2e4');

      expect(c.error, isNotNull);
      expect(c.phase, GamePhase.playing, reason: 'the game does not end');
      expect(c.engineThinking, isFalse);
    });
  });

  group('the position stays legal', () {
    /// What the backend does to every FEN it receives. A position that fails
    /// here is one the server answers with 422, stopping the game.
    void expectLegal(String fen, String context) {
      final validation = ch.Chess.validate_fen(fen);
      expect(
        validation['valid'],
        isTrue,
        reason:
            '$context produced an illegal FEN: $fen\n${validation['error']}',
      );
    }

    test('castling rights drop when a rook moves', () async {
      // The exact bug that broke the live site: the old library kept
      // announcing "KQkq" after the h1 rook had left, and python-chess
      // rejected the position with BAD_CASTLING_RIGHTS.
      final c = GameController(ScriptedEngine(['a7a6', 'b7b6', 'c7c6']))
        ..start();

      await c.playMove('h2h4');
      await c.playMove('h1h3'); // the rook leaves h1

      final fen = c.game!.fen;
      expect(
        fen.split(' ')[2],
        isNot(contains('K')),
        reason: 'white kingside castling is gone with the rook off h1',
      );
      expectLegal(fen, 'moving the h1 rook');
    });

    test('castling rights drop when the king moves', () async {
      final c = GameController(ScriptedEngine(['a7a6', 'b7b6']))..start();
      await c.playMove('e2e4');
      await c.playMove('e1e2'); // the king steps up

      final rights = c.game!.fen.split(' ')[2];
      expect(rights, isNot(contains('K')));
      expect(rights, isNot(contains('Q')));
      expectLegal(c.game!.fen, 'moving the king');
    });

    test('a promoting pawn actually becomes a piece', () async {
      // A pawn left sitting on the last rank is the other illegal position the
      // old board produced.
      final c = GameController(ScriptedEngine([]))..start();
      // Drive a white pawn to the eighth by hand.
      final game = c.game!;
      game.load('4k3/P7/8/8/8/8/8/4K3 w - - 0 1');

      await c.playMove('a7a8q');
      expect(c.game!.get('a8')?.type, ch.PieceType.QUEEN);
      expectLegal(c.game!.fen, 'promoting');
    });

    test('a full game never produces an illegal position', () async {
      // The check that would have caught the live bug on the first run: play
      // out a whole game and validate the FEN after every single move.
      final c = GameController(FirstLegalEngine())..start();

      for (var i = 0; i < 40 && c.phase == GamePhase.playing; i++) {
        final moves = c.game!.generate_moves();
        if (moves.isEmpty) break;
        final m = moves.first;
        final promo = m.promotion == null ? '' : m.promotion!.toLowerCase();
        await c.playMove(
          '${ch.Chess.algebraic(m.from)}${ch.Chess.algebraic(m.to)}$promo',
        );
        expectLegal(c.game!.fen, 'move ${i + 1}');
      }
    });
  });

  group('endings', () {
    test("fool's mate makes the user lose", () async {
      // 1. f3 e5  2. g4 Qh4#  — Black (the engine) delivers mate.
      final c = GameController(ScriptedEngine(['e7e5', 'd8h4']))..start();

      await c.playMove('f2f3');
      await c.playMove('g2g4');

      expect(c.phase, GamePhase.over);
      expect(c.result, GameResult.youLose);
      expect(c.resultDetail, 'Checkmate');
    });

    test('a king cannot be captured', () async {
      // The other failure of the old library: it allowed the king to be taken,
      // and the game continued on a position with no king at all.
      final c = GameController(ScriptedEngine([]))..start();
      c.game!.load('3k4/8/8/8/8/8/8/3QK3 w - - 0 1');

      final targets = c.game!
          .generate_moves()
          .map((m) => ch.Chess.algebraic(m.to))
          .toList();
      expect(
        targets,
        isNot(contains('d8')),
        reason: 'the square holding the black king is never a destination',
      );
    });
  });
}
