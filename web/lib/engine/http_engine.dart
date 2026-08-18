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
  /// 30 s a caldo, non 15: measured on the free tier, a hard-level search takes
  /// about 7 s, and the CPU is shared — a slow minute would otherwise surface
  /// as "could not reach the engine" when the server is simply busy.
  Duration get _timeout =>
      _cold ? const Duration(seconds: 60) : const Duration(seconds: 30);

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

  /// One POST to /move, with the network failure already turned into a
  /// sentence a player can read.
  Future<http.Response> _post(String fen, Difficulty level) async {
    try {
      final response = await _client
          .post(
            Uri.parse('$baseUrl/move'),
            headers: const {'Content-Type': 'application/json'},
            body: jsonEncode({'fen': fen, 'level': level.id}),
          )
          .timeout(_timeout);
      _cold = false;
      return response;
    } catch (_) {
      throw EngineException(
        _cold
            ? 'The engine did not answer in time. On the free server the first '
                  'wake-up can take about a minute - try again.'
            : 'Could not reach the engine. Check your connection.',
      );
    }
  }

  @override
  Future<EngineMove> bestMove(String fen, Difficulty level) async {
    var response = await _post(fen, level);

    // 503 means the engine is mid-search for someone else — it serialises
    // searches deliberately, so this is the guard working, not a fault. It
    // happens on this very page too: the wake-up ping and a first quick move
    // can overlap. One retry, honouring the server's own Retry-After, turns a
    // visible error into a slightly slower move.
    if (response.statusCode == 503) {
      final wait = int.tryParse(response.headers['retry-after'] ?? '') ?? 2;
      await Future<void>.delayed(Duration(seconds: wait.clamp(1, 5)));
      response = await _post(fen, level);
    }

    if (response.statusCode != 200) {
      // A 422 means the backend judged the position itself illegal. That is
      // never the player's doing, so the position and the server's reason go
      // to the console: without them the report is "it broke", and the actual
      // cause has to be guessed from a screenshot.
      if (response.statusCode == 422) {
        // `print`, not `debugPrint`: the latter is stripped from release
        // builds, which is exactly where this is needed — the published site.
        // ignore: avoid_print
        print('[chessbot] position rejected — FEN: $fen');
        // ignore: avoid_print
        print('[chessbot] reason: ${response.body}');
      }
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
