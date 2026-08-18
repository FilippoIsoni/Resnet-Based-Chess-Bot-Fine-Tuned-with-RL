import 'package:chess/chess.dart' as ch;
import 'package:flutter/material.dart';

import '../theme.dart';

/// The chessboard: 64 squares, drag or tap to move.
///
/// ## Why this is hand-drawn
///
/// It replaces `chess_interface`, which had to go. That package let a king be
/// captured, and never updated castling rights when a rook moved — so after a
/// few moves it produced a position chess does not allow, the backend rejected
/// it, and the game stopped with "the engine rejected the position".
///
/// The rules now come from `chess` (a port of chess.js); this widget only
/// draws them. The split is the point: no chess logic lives here beyond "which
/// squares can this piece reach", and even that is answered by the library.
class BoardView extends StatefulWidget {
  const BoardView({
    super.key,
    required this.game,
    required this.size,
    required this.onMove,
    this.interactive = true,
    this.lastMove,
  });

  final ch.Chess game;

  /// The move just played, in UCI, highlighted so the engine's reply is not
  /// missed. Following a game is much harder without it.
  final String? lastMove;
  final double size;

  /// Called with the move in UCI (`e2e4`, or `e7e8q` for a promotion).
  /// The parent applies it — this widget never mutates the game.
  final void Function(String uci) onMove;

  /// False while the engine is thinking: the board still shows, but taps do
  /// nothing. Without it the player can queue two moves in a row.
  final bool interactive;

  @override
  State<BoardView> createState() => _BoardViewState();
}

class _BoardViewState extends State<BoardView> {
  String? _selected;
  Set<String> _targets = {};

  static const _files = 'abcdefgh';

  /// Screen row/col to algebraic square. White is always at the bottom: the
  /// player is White, so the board never needs flipping.
  String _square(int row, int col) => '${_files[col]}${8 - row}';

  /// Where this piece may legally go — asked of the rules library, never
  /// worked out here.
  Set<String> _legalTargetsFrom(String square) => widget.game
      .generate_moves()
      .where((m) => ch.Chess.algebraic(m.from) == square)
      .map((m) => ch.Chess.algebraic(m.to))
      .toSet();

  void _tap(String square) {
    if (!widget.interactive) return;

    if (_selected != null && _targets.contains(square)) {
      _commit(_selected!, square);
      return;
    }

    final piece = widget.game.get(square);
    if (piece != null && piece.color == widget.game.turn) {
      setState(() {
        _selected = square;
        _targets = _legalTargetsFrom(square);
      });
      return;
    }

    setState(() {
      _selected = null;
      _targets = {};
    });
  }

  Future<void> _commit(String from, String to) async {
    // A pawn reaching the far rank MUST become another piece — chess offers no
    // option to leave it a pawn. The previous board skipped this step, which
    // is how it produced positions the backend refused.
    final piece = widget.game.get(from);
    final lastRank = to.endsWith('8') || to.endsWith('1');
    var promotion = '';

    if (piece != null && piece.type == ch.PieceType.PAWN && lastRank) {
      final chosen = await _askPromotion(piece.color == ch.Color.WHITE);
      if (chosen == null) return; // cancelled: no move at all
      promotion = chosen;
    }

    if (mounted) {
      setState(() {
        _selected = null;
        _targets = {};
      });
    }
    widget.onMove('$from$to$promotion');
  }

  Future<String?> _askPromotion(bool white) {
    return showDialog<String>(
      context: context,
      barrierDismissible: false,
      builder: (context) => AlertDialog(
        backgroundColor: surface,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(12),
          side: const BorderSide(color: outline),
        ),
        title: Text(
          'Promote to',
          style: Theme.of(context).textTheme.titleMedium,
        ),
        content: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            for (final option in _promotionOptions)
              Tooltip(
                message: option.label,
                child: InkWell(
                  onTap: () => Navigator.of(context).pop(option.code),
                  borderRadius: BorderRadius.circular(8),
                  child: Container(
                    width: 54,
                    height: 54,
                    alignment: Alignment.center,
                    margin: const EdgeInsets.symmetric(horizontal: 3),
                    decoration: BoxDecoration(
                      borderRadius: BorderRadius.circular(8),
                      border: Border.all(color: outline),
                    ),
                    child: Text(
                      option.glyph,
                      style: TextStyle(
                        fontSize: 30,
                        color: white ? pieceWhite : pieceBlack,
                        shadows: [
                          Shadow(
                            color: white ? pieceBlack : pieceWhite,
                            blurRadius: 0.8,
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final cell = widget.size / 8;

    return SizedBox(
      width: widget.size,
      height: widget.size,
      child: Column(
        // Each rank gets an explicit height. Without it the Column stretches
        // the eight rows to fill whatever vertical space it is given, and the
        // squares come out rectangular.
        mainAxisSize: MainAxisSize.min,
        children: [
          for (var row = 0; row < 8; row++)
            SizedBox(
              height: cell,
              child: Row(
                children: [
                  for (var col = 0; col < 8; col++)
                    SizedBox(
                      width: cell,
                      height: cell,
                      child: Builder(
                        builder: (_) {
                          final square = _square(row, col);
                          final piece = widget.game.get(square);
                          final last = widget.lastMove;
                          final isLast =
                              last != null &&
                              last.length >= 4 &&
                              (last.substring(0, 2) == square ||
                                  last.substring(2, 4) == square);
                          // The king in check gets its own mark: in a dark theme a
                          // check is easy to miss, and missing it loses the game.
                          final inCheck =
                              widget.game.in_check &&
                              piece != null &&
                              piece.type == ch.PieceType.KING &&
                              piece.color == widget.game.turn;
                          return _Square(
                            size: cell,
                            square: square,
                            piece: piece,
                            light: (row + col) % 2 == 0,
                            selected: _selected == square,
                            target: _targets.contains(square),
                            lastMove: isLast,
                            check: inCheck,
                            onTap: () => _tap(square),
                            onDropped: (from) => _commit(from, square),
                            draggable:
                                widget.interactive &&
                                piece != null &&
                                piece.color == widget.game.turn,
                            legalFrom: _legalTargetsFrom,
                          );
                        },
                      ),
                    ),
                ],
              ),
            ),
        ],
      ),
    );
  }
}

class _PromotionOption {
  const _PromotionOption(this.code, this.glyph, this.label);
  final String code;
  final String glyph;
  final String label;
}

const _promotionOptions = [
  _PromotionOption('q', '♛', 'Queen'),
  _PromotionOption('r', '♜', 'Rook'),
  _PromotionOption('b', '♝', 'Bishop'),
  _PromotionOption('n', '♞', 'Knight'),
];

class _Square extends StatelessWidget {
  const _Square({
    required this.size,
    required this.square,
    required this.piece,
    required this.light,
    required this.selected,
    required this.target,
    required this.lastMove,
    required this.check,
    required this.onTap,
    required this.onDropped,
    required this.draggable,
    required this.legalFrom,
  });

  final double size;
  final String square;
  final ch.Piece? piece;
  final bool light;
  final bool selected;
  final bool target;
  final bool lastMove;
  final bool check;
  final VoidCallback onTap;
  final void Function(String from) onDropped;
  final bool draggable;
  final Set<String> Function(String) legalFrom;

  @override
  Widget build(BuildContext context) {
    final base = light ? boardLight : boardDark;
    final currentPiece = piece;

    Widget content = Container(
      width: size,
      height: size,
      color: check
          ? Color.alphaBlend(danger.withValues(alpha: 0.55), base)
          : selected
          ? Color.alphaBlend(accent.withValues(alpha: 0.30), base)
          : lastMove
          ? Color.alphaBlend(accent.withValues(alpha: 0.14), base)
          : base,
      child: Stack(
        alignment: Alignment.center,
        children: [
          // A legal destination: a dot on an empty square, a ring on a
          // capture. Quieter than tinting the whole square, and read instantly.
          if (target)
            currentPiece == null
                ? Container(
                    width: size * 0.26,
                    height: size * 0.26,
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      color: accent.withValues(alpha: 0.45),
                    ),
                  )
                : Container(
                    width: size * 0.86,
                    height: size * 0.86,
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      border: Border.all(
                        color: accent.withValues(alpha: 0.75),
                        width: 3,
                      ),
                    ),
                  ),
          if (currentPiece != null)
            _PieceGlyph(piece: currentPiece, size: size),
        ],
      ),
    );

    if (currentPiece != null && draggable) {
      content = Draggable<String>(
        data: square,
        feedback: _PieceGlyph(piece: currentPiece, size: size * 1.1),
        childWhenDragging: SizedBox(
          width: size,
          height: size,
          child: ColoredBox(color: base),
        ),
        child: content,
      );
    }

    return DragTarget<String>(
      onWillAcceptWithDetails: (d) => legalFrom(d.data).contains(square),
      onAcceptWithDetails: (d) => onDropped(d.data),
      builder: (context, _, _) => GestureDetector(onTap: onTap, child: content),
    );
  }
}

/// A piece drawn as a Unicode glyph.
///
/// No image assets: the figurine characters live in every system font, they
/// scale without blurring, and they cost nothing in the bundle.
class _PieceGlyph extends StatelessWidget {
  const _PieceGlyph({required this.piece, required this.size});

  final ch.Piece piece;
  final double size;

  // The solid (nominally black) glyphs are used for both colours, with fill
  // and outline telling them apart: the hollow "white" characters render
  // inconsistently across platforms and disappear on a dark board.
  static const _glyphs = {
    'p': '♟',
    'n': '♞',
    'b': '♝',
    'r': '♜',
    'q': '♛',
    'k': '♚',
  };

  @override
  Widget build(BuildContext context) {
    final white = piece.color == ch.Color.WHITE;
    final glyph = _glyphs[piece.type.toLowerCase()] ?? '?';

    return SizedBox(
      width: size,
      height: size,
      // FittedBox, not a bare Text: chess glyphs have generous ascenders and
      // some fonts lay them out taller than the declared size, which pushes
      // the row past its height and turns the squares into rectangles. This
      // scales whatever the font produces into the square it is given.
      child: FittedBox(
        fit: BoxFit.contain,
        child: Text(
          glyph,
          style: TextStyle(
            fontSize: size * 0.86,
            height: 1.0,
            color: white ? pieceWhite : pieceBlack,
            shadows: [
              Shadow(color: white ? pieceBlack : pieceWhite, blurRadius: 0.8),
            ],
          ),
        ),
      ),
    );
  }
}
