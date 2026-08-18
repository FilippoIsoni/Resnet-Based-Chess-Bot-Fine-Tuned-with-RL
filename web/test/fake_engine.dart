import 'dart:math';

import 'package:chess/chess.dart' as ch;
import 'package:chessbot_ui/engine/engine.dart';

/// A stand-in opponent that plays a random legal move. **Test-only.**
///
/// It exists so widget tests can mount the whole app without a network: a test
/// that needs a running server fails for reasons that have nothing to do with
/// the code under test.
class FakeEngine implements Engine {
  FakeEngine({int? seed}) : _random = Random(seed);

  final Random _random;

  @override
  Future<bool> ping() async => true;

  @override
  Future<EngineMove> bestMove(String fen, Difficulty level) async {
    final game = ch.Chess.fromFEN(fen);
    final moves = game.generate_moves();
    if (moves.isEmpty) {
      throw const EngineException('No legal move in this position.');
    }

    final m = moves[_random.nextInt(moves.length)];
    final promotion = m.promotion == null ? '' : m.promotion!.toLowerCase();

    return EngineMove(
      uci: '${ch.Chess.algebraic(m.from)}${ch.Chess.algebraic(m.to)}$promotion',
      // Random but damped, so the end-of-game chart has something plausible
      // to draw without looking unhinged.
      evaluation: (_random.nextDouble() - 0.5) * 1.2,
    );
  }
}
