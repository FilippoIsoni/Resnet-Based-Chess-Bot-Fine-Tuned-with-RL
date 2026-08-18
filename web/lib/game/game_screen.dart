import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../engine/engine.dart';
import '../theme.dart';
import 'board_view.dart';
import 'evaluation_chart.dart';
import 'game_controller.dart';

/// The app's only screen. It changes face according to the phase.
class GameScreen extends StatelessWidget {
  const GameScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final controller = context.watch<GameController>();

    return Scaffold(
      body: SafeArea(
        child: Center(
          child: SingleChildScrollView(
            padding: const EdgeInsets.all(24),
            child: ConstrainedBox(
              constraints: const BoxConstraints(maxWidth: 520),
              // Screens cross-fade and drift up slightly instead of snapping.
              // The `ValueKey` on the phase is what makes the switcher treat
              // them as different children — without it the content changes in
              // place and nothing animates.
              child: AnimatedSwitcher(
                duration: const Duration(milliseconds: 280),
                switchInCurve: Curves.easeOutCubic,
                switchOutCurve: Curves.easeIn,
                transitionBuilder: (child, animation) => FadeTransition(
                  opacity: animation,
                  child: SlideTransition(
                    position: Tween(
                      begin: const Offset(0, 0.03),
                      end: Offset.zero,
                    ).animate(animation),
                    child: child,
                  ),
                ),
                child: switch (controller.phase) {
                  GamePhase.menu => const _MenuView(key: ValueKey('menu')),
                  GamePhase.playing => const _PlayingView(
                    key: ValueKey('play'),
                  ),
                  GamePhase.over => const _GameOverView(key: ValueKey('over')),
                },
              ),
            ),
          ),
        ),
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Menu
// ---------------------------------------------------------------------------

class _MenuView extends StatelessWidget {
  const _MenuView({super.key});

  @override
  Widget build(BuildContext context) {
    final controller = context.watch<GameController>();
    final theme = Theme.of(context);

    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Text('Play the network', style: theme.textTheme.headlineMedium),
        const SizedBox(height: 8),
        Text(
          'A neural network trained from scratch on Lichess games, '
          'without copying from other engines.',
          textAlign: TextAlign.center,
          style: theme.textTheme.bodyMedium,
        ),
        const SizedBox(height: 32),
        Align(
          alignment: Alignment.centerLeft,
          child: Text('Difficulty', style: theme.textTheme.titleMedium),
        ),
        const SizedBox(height: 12),
        ...Difficulty.values.map(
          (level) => _DifficultyTile(
            level: level,
            selected: controller.difficulty == level,
            onTap: () => controller.setDifficulty(level),
          ),
        ),
        const SizedBox(height: 24),
        SizedBox(
          width: double.infinity,
          child: FilledButton(
            onPressed: controller.start,
            style: FilledButton.styleFrom(
              padding: const EdgeInsets.symmetric(vertical: 16),
            ),
            child: const Text('Start game'),
          ),
        ),
        const SizedBox(height: 12),
        Text(
          'You play White and move first.',
          style: theme.textTheme.bodySmall,
        ),
      ],
    );
  }
}

class _DifficultyTile extends StatelessWidget {
  const _DifficultyTile({
    required this.level,
    required this.selected,
    required this.onTap,
  });

  final Difficulty level;
  final bool selected;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Material(
        color: selected ? accent.withValues(alpha: 0.10) : surface,
        borderRadius: BorderRadius.circular(10),
        child: InkWell(
          onTap: onTap,
          borderRadius: BorderRadius.circular(10),
          child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(10),
              border: Border.all(
                color: selected ? accent : outline,
                width: selected ? 1.5 : 1,
              ),
            ),
            child: Row(
              children: [
                Icon(
                  selected
                      ? Icons.radio_button_checked
                      : Icons.radio_button_off,
                  color: selected ? accent : outline,
                  size: 20,
                ),
                const SizedBox(width: 12),
                Text(
                  level.label,
                  style: TextStyle(
                    fontWeight: FontWeight.w500,
                    color: selected ? textPrimary : textSecondary,
                  ),
                ),
                const Spacer(),
                Text(
                  level.strength,
                  style: selected
                      ? monoLabel.copyWith(color: accentSoft)
                      : monoLabel,
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Playing
// ---------------------------------------------------------------------------

class _PlayingView extends StatelessWidget {
  const _PlayingView({super.key});

  @override
  Widget build(BuildContext context) {
    final controller = context.watch<GameController>();
    final game = controller.game;
    if (game == null) return const SizedBox.shrink();

    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Row(
          children: [
            Text(
              controller.difficulty.label.toUpperCase(),
              style: monoLabel.copyWith(color: accentSoft),
            ),
            const Spacer(),
            TextButton.icon(
              onPressed: controller.restart,
              icon: const Icon(Icons.close, size: 18),
              label: const Text('Resign'),
            ),
          ],
        ),
        const SizedBox(height: 8),
        LayoutBuilder(
          builder: (context, constraints) {
            final size = constraints.maxWidth.clamp(280.0, 460.0);
            // Centred inside a box of exactly that side: inside a Column the
            // builder gets a bounded width but unbounded height, and without
            // this the board is stretched vertically into rectangles.
            return Center(
              child: BoardView(
                game: game,
                size: size,
                // While the engine thinks the board is inert: without this the
                // player can queue two moves in a row.
                interactive: !controller.engineThinking,
                lastMove: controller.lastMove,
                onMove: controller.playMove,
              ),
            );
          },
        ),
        const SizedBox(height: 16),
        // The status line changes on every move, so it cross-fades rather than
        // flicking between states.
        AnimatedSwitcher(
          duration: const Duration(milliseconds: 200),
          child: _StatusBar(
            key: ValueKey(
              controller.error ??
                  (controller.engineThinking ? 'think' : 'idle'),
            ),
            thinking: controller.engineThinking,
            error: controller.error,
          ),
        ),
      ],
    );
  }
}

/// The line under the board: whose turn it is, or what went wrong.
class _StatusBar extends StatelessWidget {
  const _StatusBar({super.key, required this.thinking, required this.error});

  final bool thinking;
  final String? error;

  @override
  Widget build(BuildContext context) {
    if (error != null) {
      return Container(
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(
          color: danger.withValues(alpha: 0.12),
          borderRadius: BorderRadius.circular(8),
          border: Border.all(color: danger.withValues(alpha: 0.35)),
        ),
        child: Row(
          children: [
            const Icon(Icons.error_outline, color: danger, size: 20),
            const SizedBox(width: 10),
            Expanded(
              child: Text(
                error!,
                style: const TextStyle(color: textSecondary, fontSize: 13),
              ),
            ),
          ],
        ),
      );
    }

    if (thinking) {
      return const Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          SizedBox(
            width: 14,
            height: 14,
            child: CircularProgressIndicator(strokeWidth: 2),
          ),
          SizedBox(width: 12),
          Text('Thinking...', style: TextStyle(color: textSecondary)),
        ],
      );
    }

    return const Text('Your move.', style: TextStyle(color: textSecondary));
  }
}

// ---------------------------------------------------------------------------
// Game over
// ---------------------------------------------------------------------------

class _GameOverView extends StatelessWidget {
  const _GameOverView({super.key});

  @override
  Widget build(BuildContext context) {
    final controller = context.watch<GameController>();
    final theme = Theme.of(context);
    final result = controller.result;

    return Column(
      mainAxisSize: MainAxisSize.min,
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Container(
          padding: const EdgeInsets.all(24),
          decoration: BoxDecoration(
            color: surface,
            borderRadius: BorderRadius.circular(12),
            border: Border.all(color: outline),
          ),
          child: Column(
            children: [
              Icon(
                switch (result) {
                  GameResult.youWin => Icons.emoji_events_outlined,
                  GameResult.youLose => Icons.flag_outlined,
                  _ => Icons.remove_outlined,
                },
                size: 40,
                color: result == GameResult.youWin ? accent : textMuted,
              ),
              const SizedBox(height: 12),
              Text(result?.label ?? '', style: theme.textTheme.headlineSmall),
              if (controller.resultDetail != null) ...[
                const SizedBox(height: 6),
                Text(
                  controller.resultDetail!,
                  textAlign: TextAlign.center,
                  style: theme.textTheme.bodyMedium,
                ),
              ],
              const SizedBox(height: 10),
              Text(controller.difficulty.label.toUpperCase(), style: monoLabel),
            ],
          ),
        ),
        if (controller.evaluations.length >= 2) ...[
          const SizedBox(height: 16),
          Container(
            padding: const EdgeInsets.all(20),
            decoration: BoxDecoration(
              color: surface,
              borderRadius: BorderRadius.circular(12),
              border: Border.all(color: outline),
            ),
            child: EvaluationChart(evaluations: controller.evaluations),
          ),
        ],
        const SizedBox(height: 20),
        FilledButton(
          onPressed: controller.restart,
          style: FilledButton.styleFrom(
            padding: const EdgeInsets.symmetric(vertical: 16),
          ),
          child: const Text('Play again'),
        ),
      ],
    );
  }
}
