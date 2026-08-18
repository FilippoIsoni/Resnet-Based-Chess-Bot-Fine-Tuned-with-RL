import 'package:chess_interface/providers/chess_board_provider.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'config.dart';
import 'engine/engine.dart';
import 'engine/http_engine.dart';
import 'game/game_controller.dart';
import 'game/game_screen.dart';
import 'theme.dart';

void main() {
  final engine = HttpEngine(backendUrl);

  // The free Space sleeps after a spell of inactivity. Calling it now starts
  // the wake-up while the user reads the page and picks a difficulty, instead
  // of making them wait half a minute on the first move. Deliberately not
  // awaited: if it fails, the first real request will surface the problem.
  engine.ping();

  runApp(ChessBotApp(engine: engine));
}

class ChessBotApp extends StatelessWidget {
  const ChessBotApp({super.key, required this.engine});

  final Engine engine;

  @override
  Widget build(BuildContext context) {
    // Required by `ChessBoardWidget`, which wraps itself in a
    // `Consumer<ChessBoardProvider>`. It does not hold the game state — that
    // lives in `GameController` — it is only the signal to repaint the board
    // after the engine moves.
    final boardProvider = ChessBoardProvider();

    return MultiProvider(
      providers: [
        ChangeNotifierProvider(
          create: (_) => GameController(engine, onBoardChanged: boardProvider.notify),
        ),
        ChangeNotifierProvider.value(value: boardProvider),
      ],
      child: MaterialApp(
        title: 'Chess bot',
        debugShowCheckedModeBanner: false,
        theme: buildDarkTheme(),
        home: const GameScreen(),
      ),
    );
  }
}
