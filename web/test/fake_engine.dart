import 'dart:math';

import 'package:chess_interface_dart/logical_interface/interface.dart';
import 'package:chess_interface_dart/logical_interface/piece.dart';
import 'package:chess_interface_dart/logical_interface/position.dart';

import 'package:chessbot_ui/engine/engine.dart';
import 'package:chessbot_ui/engine/uci.dart';

/// A stand-in opponent that plays at random. **Test-only.**
///
/// It used to back the app itself, while the backend was being written by
/// someone else. Now that the real engine exists the app always talks to it,
/// and this lives here purely so widget tests can mount the whole app without
/// a network: a test that depends on a running server is a test that fails for
/// reasons that have nothing to do with the code.
///
/// It picks a random legal move and returns an invented but plausible
/// evaluation, so the end-of-game chart has something to draw.
class FakeEngine implements Engine {
  FakeEngine({int? seed}) : _random = Random(seed);

  final Random _random;

  @override
  Future<bool> ping() async => true;

  @override
  Future<EngineMove> bestMove(String fen, Difficulty level) async {
    // A fake delay proportional to the level: without it the interface feels
    // instant and the "thinking" state never shows, which it certainly will
    // against the real backend.
    await Future<void>.delayed(
      Duration(milliseconds: 200 + level.simulations ~/ 2),
    );

    final game = ChessBoardInterface(fen: fen);
    final moves = _allLegalMoves(game);

    if (moves.isEmpty) {
      throw const EngineException('No legal move in this position.');
    }

    final move = moves[_random.nextInt(moves.length)];
    return EngineMove(
      uci: move,
      // Random but damped, so the chart wobbles without looking unhinged.
      evaluation: (_random.nextDouble() - 0.5) * 1.2,
      thinkingMs: 200 + level.simulations ~/ 2,
    );
  }

  /// Every legal move in the position, in UCI.
  ///
  /// Brute force: 64 origin squares times 64 destinations. Wasteful, but this
  /// is a stand-in and the library exposes no move generator.
  List<String> _allLegalMoves(ChessBoardInterface game) {
    final moves = <String>[];

    for (var fromRow = 0; fromRow < 8; fromRow++) {
      for (var fromCol = 0; fromCol < 8; fromCol++) {
        final from = Position(row: fromRow, col: fromCol);
        final piece = game.getPiece(from);
        if (piece == null || piece.color != game.turn) continue;

        for (var toRow = 0; toRow < 8; toRow++) {
          for (var toCol = 0; toCol < 8; toCol++) {
            final to = Position(row: toRow, col: toCol);

            // The only way to know a move is legal is to try it on a copy:
            // `MoveValidator` does not cover castling or discovered checks.
            final copy = ChessBoardInterface(fen: game.toFEN());
            if (!copy.move(from, to)) continue;

            // Promotions must be declared: the real engine would send `e7e8q`.
            final promotes = piece.type == PieceType.pawn &&
                (toRow == 0 || toRow == 7);
            moves.add(positionsToUci(from, to, promotion: promotes ? 'q' : null));
          }
        }
      }
    }
    return moves;
  }
}
