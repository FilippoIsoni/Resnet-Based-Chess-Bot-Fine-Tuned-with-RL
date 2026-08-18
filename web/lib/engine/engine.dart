/// What the UI expects from an engine, without knowing where it runs.
///
/// There are two implementations: [FakeEngine] plays randomly in the browser
/// and exists so the interface can be built without a backend, [HttpEngine]
/// talks to the real server. The UI cannot tell which one it has.
library;

/// The three difficulty levels.
///
/// They are the **same engine** at different search depths: more simulations
/// means more lines checked before committing. There is no weaker network for
/// the easy level.
enum Difficulty {
  easy('easy', 'Casual', '~1500 Elo', 50),
  medium('medium', 'Club', '1964 Elo', 200),
  hard('hard', 'Strong', '~2100 Elo', 800);

  const Difficulty(this.id, this.label, this.strength, this.simulations);

  /// What the backend calls it (see `docs/API_CONTRACT.md`).
  final String id;

  /// What we call it to the user.
  final String label;

  /// Indicative strength. Only `medium` is actually measured, against
  /// Stockfish; the other two are estimates — hence the tilde in the UI.
  final String strength;

  /// MCTS simulations. Shown for interest only; the backend decides its own.
  final int simulations;
}

/// A move chosen by the engine.
class EngineMove {
  const EngineMove({
    required this.uci,
    required this.evaluation,
    this.thinkingMs = 0,
  });

  /// The move in UCI notation: `e2e4`, or `e7e8q` for a promotion.
  final String uci;

  /// How well the engine thinks the position is going, from -1 to +1.
  ///
  /// **From White's point of view**, always: +1 White wins, -1 Black wins. The
  /// backend sends it from the side-to-move's perspective, and [HttpEngine]
  /// converts it once, at parse time.
  ///
  /// Why pin the convention down here: the sign crosses three boundaries
  /// (engine, JSON, interface) and getting it wrong raises no error at all —
  /// just a chart that tells the game backwards.
  final double evaluation;

  /// Milliseconds spent thinking.
  final int thinkingMs;
}

/// An engine error worth showing to the user.
class EngineException implements Exception {
  const EngineException(this.message);

  /// A message already written for a human, not an HTTP code.
  final String message;

  @override
  String toString() => message;
}

/// An opponent that can choose a move.
abstract class Engine {
  /// The engine's move in the given position.
  ///
  /// Throws [EngineException] if it cannot be obtained.
  Future<EngineMove> bestMove(String fen, Difficulty level);

  /// Wakes the engine and reports whether it answers.
  ///
  /// The free server goes to sleep: calling this when the page opens starts
  /// the wake-up while the user picks a difficulty, instead of making them
  /// wait 30 seconds on the first click.
  Future<bool> ping();
}
