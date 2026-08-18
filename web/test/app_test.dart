/// The app starts and reaches the board.
///
/// This exists because of a real bug: `ChessBoardWidget` wraps itself in a
/// `Consumer<ChessBoardProvider>`, and without that provider in the tree the
/// app showed a red error screen instead of the board. `flutter analyze` could
/// not catch it — the provider is looked up at runtime — and the controller
/// tests could not either, because they mount no widgets.
///
/// These are few and coarse on purpose: they check that the three screens
/// mount without blowing up, not how they look.
library;

import 'package:chessbot_ui/engine/engine.dart';
import 'package:chessbot_ui/main.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'fake_engine.dart';

void main() {
  testWidgets('the opening menu shows up', (tester) async {
    await tester.pumpWidget(ChessBotApp(engine: FakeEngine(seed: 1)));

    expect(find.text('Start game'), findsOneWidget);
    for (final level in Difficulty.values) {
      expect(find.text(level.label), findsOneWidget);
    }
  });

  testWidgets('pressing start reaches the board', (tester) async {
    await tester.pumpWidget(ChessBotApp(engine: FakeEngine(seed: 1)));

    await tester.tap(find.text('Start game'));
    await tester.pumpAndSettle();

    // With `ChessBoardProvider` missing there is an ErrorWidget here instead
    // of the board: that is exactly the case this test guards.
    expect(tester.takeException(), isNull);
    expect(find.byType(GridView), findsOneWidget);
    expect(find.text('Your move.'), findsOneWidget);
  });

  testWidgets('a different level can be chosen', (tester) async {
    await tester.pumpWidget(ChessBotApp(engine: FakeEngine(seed: 1)));

    await tester.tap(find.text(Difficulty.hard.label));
    await tester.pump();
    await tester.tap(find.text('Start game'));
    await tester.pumpAndSettle();

    // On the board the level is shown in upper case.
    expect(find.text(Difficulty.hard.label.toUpperCase()), findsOneWidget);
    expect(tester.takeException(), isNull);
  });

  testWidgets('resigning returns to the menu', (tester) async {
    await tester.pumpWidget(ChessBotApp(engine: FakeEngine(seed: 1)));

    await tester.tap(find.text('Start game'));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Resign'));
    await tester.pumpAndSettle();

    expect(find.text('Start game'), findsOneWidget);
    expect(tester.takeException(), isNull);
  });
}
