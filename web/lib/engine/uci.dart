/// Translation between the engine's UCI notation and board coordinates.
///
/// ## The two conventions
///
/// The engine speaks **UCI**: `e2e4` means "from square e2 to e4", and `e7e8q`
/// appends the promotion piece.
///
/// The board library uses **row and column**, both 0 to 7, with row 0 at the
/// top:
///
///     row 0  ->  rank 8   (where Black starts)
///     row 7  ->  rank 1   (where White starts)
///     col 0  ->  file a
///     col 7  ->  file h
///
/// So `row = 8 - rank`, not `rank - 1`. That is the same formula the package
/// uses internally for en passant, so the convention is confirmed by its code
/// rather than guessed.
///
/// ## Why this has its own file
///
/// If this conversion is wrong the engine plays legal moves that are not the
/// ones it chose — mirrored vertically. Nothing crashes, and on the board it
/// merely looks like it plays badly. Isolating it makes it testable in two
/// lines: see `test/uci_test.dart`.
library;

import 'package:chess_interface_dart/logical_interface/piece.dart';
import 'package:chess_interface_dart/logical_interface/position.dart';

/// From a UCI square (`e4`) to row and column.
Position squareToPosition(String square) {
  if (square.length != 2) {
    throw FormatException('Not a UCI square: "$square"');
  }

  final col = square.codeUnitAt(0) - 'a'.codeUnitAt(0);
  final rank = int.tryParse(square[1]);

  if (col < 0 || col > 7 || rank == null || rank < 1 || rank > 8) {
    throw FormatException('UCI square off the board: "$square"');
  }

  return Position(row: 8 - rank, col: col);
}

/// From row and column to a UCI square (`e4`).
String positionToSquare(Position position) {
  final file = String.fromCharCode('a'.codeUnitAt(0) + position.col);
  final rank = 8 - position.row;
  return '$file$rank';
}

/// The origin square of a UCI move.
Position uciFrom(String uci) => squareToPosition(uci.substring(0, 2));

/// The destination square of a UCI move.
Position uciTo(String uci) => squareToPosition(uci.substring(2, 4));

/// The promotion piece of a UCI move, if any.
///
/// `e7e8q` -> queen. A normal move has no fifth character and yields `null`.
PieceType? uciPromotion(String uci) {
  if (uci.length < 5) return null;

  switch (uci[4].toLowerCase()) {
    case 'q':
      return PieceType.queen;
    case 'r':
      return PieceType.rook;
    case 'b':
      return PieceType.bishop;
    case 'n':
      return PieceType.knight;
    default:
      return null;
  }
}

/// Builds a UCI move from two positions.
String positionsToUci(Position from, Position to, {String? promotion}) {
  return '${positionToSquare(from)}${positionToSquare(to)}${promotion ?? ''}';
}
