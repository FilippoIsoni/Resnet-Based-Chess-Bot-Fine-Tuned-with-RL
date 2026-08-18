/// Conversion between UCI and board coordinates.
///
/// The most important test in the app. If this conversion is wrong the engine
/// plays legal but **mirrored** moves: nothing crashes, and on the board it
/// merely looks like it plays badly. Two lines catch it here.
library;

import 'package:chess_interface_dart/logical_interface/piece.dart';
import 'package:chess_interface_dart/logical_interface/position.dart';
import 'package:chessbot_ui/engine/uci.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('UCI to coordinates', () {
    test('a1 is bottom-left', () {
      // Rank 1 is row 7 because row 0 is the top of the board.
      final pos = squareToPosition('a1');
      expect(pos.row, 7);
      expect(pos.col, 0);
    });

    test('h8 is top-right', () {
      final pos = squareToPosition('h8');
      expect(pos.row, 0);
      expect(pos.col, 7);
    });

    test('e4 lands where expected', () {
      final pos = squareToPosition('e4');
      expect(pos.row, 4);
      expect(pos.col, 4);
    });

    test('a made-up square is rejected', () {
      expect(() => squareToPosition('z9'), throwsFormatException);
      expect(() => squareToPosition('e'), throwsFormatException);
      expect(() => squareToPosition('e0'), throwsFormatException);
    });
  });

  group('round trip', () {
    test('every square on the board comes back as itself', () {
      // The check that matters: if the formula were inverted in one
      // direction only, this loop would fail on half the board.
      for (final file in 'abcdefgh'.split('')) {
        for (var rank = 1; rank <= 8; rank++) {
          final square = '$file$rank';
          expect(positionToSquare(squareToPosition(square)), square);
        }
      }
    });
  });

  group('whole moves', () {
    test('e2e4 splits into its two squares', () {
      expect(uciFrom('e2e4'), Position(row: 6, col: 4));
      expect(uciTo('e2e4'), Position(row: 4, col: 4));
      expect(uciPromotion('e2e4'), isNull);
    });

    test('a promotion carries its piece', () {
      expect(uciPromotion('e7e8q'), PieceType.queen);
      expect(uciPromotion('a2a1n'), PieceType.knight);
      expect(uciPromotion('b7b8r'), PieceType.rook);
      expect(uciPromotion('c2c1b'), PieceType.bishop);
    });

    test('the original string is rebuilt', () {
      expect(
        positionsToUci(uciFrom('g1f3'), uciTo('g1f3')),
        'g1f3',
      );
      expect(
        positionsToUci(uciFrom('e7e8q'), uciTo('e7e8q'), promotion: 'q'),
        'e7e8q',
      );
    });
  });
}
