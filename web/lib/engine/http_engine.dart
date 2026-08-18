import 'dart:convert';

import 'package:http/http.dart' as http;

import 'engine.dart';

/// The real engine, reached over HTTP.
///
/// Implements the contract in `docs/API_CONTRACT.md`. If that document
/// changes, this file changes with it.
class HttpEngine implements Engine {
  HttpEngine(this.baseUrl, {http.Client? client})
      : _client = client ?? http.Client();

  /// No trailing slash, e.g. `https://someone-chessbot.hf.space`.
  final String baseUrl;
  final http.Client _client;

  /// True until the first response arrives.
  ///
  /// Decides how long to wait: the free server sleeps, and waking it takes
  /// about half a minute — but only the first time.
  bool _cold = true;

  /// How long to wait for a move.
  ///
  /// The first request may have to wake the server; later ones do not. A
  /// single short timeout would fail every time on open; a single long one
  /// would leave the user staring at a frozen board when the server really is
  /// unreachable.
  Duration get _timeout =>
      _cold ? const Duration(seconds: 45) : const Duration(seconds: 15);

  @override
  Future<bool> ping() async {
    try {
      final response = await _client
          .get(Uri.parse('$baseUrl/health'))
          .timeout(const Duration(seconds: 40));

      if (response.statusCode != 200) return false;

      final body = jsonDecode(response.body) as Map<String, dynamic>;
      final loaded = body['model_loaded'] == true;
      if (loaded) _cold = false;
      return loaded;
    } catch (_) {
      // A failed ping is not an error worth showing: it only means the server
      // is still asleep, and the UI phrases that its own way.
      return false;
    }
  }

  @override
  Future<EngineMove> bestMove(String fen, Difficulty level) async {
    final http.Response response;
    try {
      response = await _client
          .post(
            Uri.parse('$baseUrl/move'),
            headers: const {'Content-Type': 'application/json'},
            body: jsonEncode({'fen': fen, 'level': level.id}),
          )
          .timeout(_timeout);
    } catch (_) {
      throw EngineException(
        _cold
            ? 'The engine did not answer in time. On the free server the first '
                'wake-up can take about thirty seconds - try again.'
            : 'Could not reach the engine. Check your connection.',
      );
    }

    _cold = false;

    if (response.statusCode != 200) {
      throw EngineException(_messageFor(response));
    }

    final body = jsonDecode(response.body) as Map<String, dynamic>;
    final uci = body['move'] as String?;

    if (uci == null) {
      throw const EngineException('This position is already finished.');
    }

    return EngineMove(
      uci: uci,
      evaluation: _whitePovEvaluation(body['eval'], fen),
      thinkingMs: (body['ms'] as num?)?.toInt() ?? 0,
    );
  }

  /// Converts the evaluation to White's point of view.
  ///
  /// **The backend sends it from the side-to-move's perspective** (negamax
  /// convention, the same as the MCTS): +0.5 means "whoever is to move is
  /// better". If Black is to move, that is -0.5 for White.
  ///
  /// Without this conversion the end-of-game chart would flip sign on every
  /// move and look as though the game turned constantly. This is the only
  /// place in the app where the sign is touched.
  static double _whitePovEvaluation(Object? raw, String fen) {
    final value = (raw as num?)?.toDouble() ?? 0.0;
    final blackToMove = fen.split(' ').elementAtOrNull(1) == 'b';
    return blackToMove ? -value : value;
  }

  /// Turns an HTTP failure into a sentence for the user.
  static String _messageFor(http.Response response) {
    switch (response.statusCode) {
      case 422:
        return 'The engine rejected the position. That is a bug in this app, '
            'not something you did.';
      case 429:
        return 'Too many requests in a short time. Wait a few seconds.';
      case 503:
        return 'The engine is busy with other games. Try again in a moment.';
      default:
        return 'The engine returned an error (${response.statusCode}).';
    }
  }
}
