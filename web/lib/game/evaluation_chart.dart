import 'package:flutter/material.dart';

import '../theme.dart';

/// How the game swung, shown once it is over.
///
/// A line that rises while you were better and falls while you were worse,
/// with the moment things turned marked on it.
///
/// ## Why it is hand-drawn
///
/// A charting library brings dozens of options nobody needs and a file nobody
/// reads. This is thirty lines of `Canvas`: it can be understood at a glance
/// and it does exactly one thing.
///
/// The values already arrive from White's point of view (see `EngineMove`) and
/// the user plays White, so **above the midline means the user was better**,
/// with no further conversion.
class EvaluationChart extends StatelessWidget {
  const EvaluationChart({super.key, required this.evaluations});

  final List<double> evaluations;

  @override
  Widget build(BuildContext context) {
    if (evaluations.length < 2) {
      return const SizedBox.shrink();
    }

    final turningPoint = _findTurningPoint(evaluations);
    final theme = Theme.of(context);

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text('How the game swung', style: theme.textTheme.titleMedium),
        const SizedBox(height: 4),
        Text(
          'The line rises while you were the one doing better.',
          style: theme.textTheme.bodySmall,
        ),
        const SizedBox(height: 12),
        // The curve draws itself left to right the first time it appears. It
        // makes the chart read as a sequence of moves rather than a static
        // shape — the one place in the app where an animation carries meaning
        // instead of decoration.
        TweenAnimationBuilder<double>(
          tween: Tween(begin: 0, end: 1),
          duration: const Duration(milliseconds: 700),
          curve: Curves.easeOutCubic,
          builder: (context, progress, _) => SizedBox(
            height: 140,
            width: double.infinity,
            child: CustomPaint(
              painter: _ChartPainter(
                evaluations: evaluations,
                turningPoint: turningPoint,
                progress: progress,
              ),
            ),
          ),
        ),
        if (turningPoint != null) ...[
          const SizedBox(height: 12),
          Row(
            children: [
              const Icon(Icons.trending_down, size: 18, color: accentSoft),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  'Move ${turningPoint + 1} is where it turned.',
                  style: theme.textTheme.bodyMedium,
                ),
              ),
            ],
          ),
        ],
      ],
    );
  }

  /// The move after which the evaluation dropped most for the user.
  ///
  /// This is not real analysis — we do not know what the better move was — but
  /// it answers the question a player actually asks: "where did it go wrong?".
  /// Returns `null` when there is no clear drop, so a level game is not given
  /// an invented dramatic moment.
  static int? _findTurningPoint(List<double> values) {
    if (values.length < 3) return null;

    var worstDrop = 0.0;
    int? index;

    for (var i = 1; i < values.length; i++) {
      final drop = values[i - 1] - values[i];
      if (drop > worstDrop) {
        worstDrop = drop;
        index = i;
      }
    }

    // Below this threshold the "drop" is evaluation noise, not a mistake.
    return worstDrop >= 0.4 ? index : null;
  }
}

class _ChartPainter extends CustomPainter {
  _ChartPainter({
    required this.evaluations,
    required this.turningPoint,
    this.progress = 1,
  });

  final List<double> evaluations;
  final int? turningPoint;

  /// How much of the curve to draw, 0 to 1. Used for the reveal animation.
  final double progress;

  @override
  void paint(Canvas canvas, Size size) {
    // The scale is fixed from -1 to +1, like the engine's evaluation. Fitting
    // it to the data would make a tiny edge look like a rout.
    double y(double evaluation) => size.height * (1 - (evaluation + 1) / 2);
    double x(int index) =>
        evaluations.length == 1 ? 0 : size.width * index / (evaluations.length - 1);

    // Two barely-there bands: above the midline is your ground, below is the
    // engine's. They make the chart readable without axis labels.
    canvas.drawRect(
      Rect.fromLTWH(0, 0, size.width, size.height / 2),
      Paint()..color = accent.withValues(alpha: 0.07),
    );
    canvas.drawRect(
      Rect.fromLTWH(0, size.height / 2, size.width, size.height / 2),
      Paint()..color = danger.withValues(alpha: 0.09),
    );

    // The equality line.
    canvas.drawLine(
      Offset(0, size.height / 2),
      Offset(size.width, size.height / 2),
      Paint()
        ..color = outline
        ..strokeWidth = 1,
    );

    // The curve.
    final path = Path()..moveTo(x(0), y(evaluations.first));
    for (var i = 1; i < evaluations.length; i++) {
      path.lineTo(x(i), y(evaluations[i]));
    }

    // Trim the path by actual traced length rather than by point count, so the
    // reveal moves at a steady speed instead of jumping between vertices.
    final stroke = Paint()
      ..color = accent
      ..strokeWidth = 2.5
      ..style = PaintingStyle.stroke
      ..strokeJoin = StrokeJoin.round
      ..strokeCap = StrokeCap.round;

    if (progress >= 1) {
      canvas.drawPath(path, stroke);
    } else {
      for (final metric in path.computeMetrics()) {
        canvas.drawPath(
          metric.extractPath(0, metric.length * progress.clamp(0, 1)),
          stroke,
        );
      }
    }

    // The turning point. It appears only once the curve has reached it.
    final revealedUpTo = (evaluations.length - 1) * progress;
    if (turningPoint != null && turningPoint! <= revealedUpTo) {
      final point = Offset(x(turningPoint!), y(evaluations[turningPoint!]));
      canvas.drawCircle(point, 5, Paint()..color = accentSoft);
      canvas.drawCircle(
        point,
        5,
        Paint()
          ..color = background
          ..strokeWidth = 2
          ..style = PaintingStyle.stroke,
      );
    }
  }

  @override
  bool shouldRepaint(_ChartPainter old) =>
      old.evaluations != evaluations ||
      old.turningPoint != turningPoint ||
      old.progress != progress;
}
