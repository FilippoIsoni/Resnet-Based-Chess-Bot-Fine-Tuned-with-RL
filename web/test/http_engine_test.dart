/// The HTTP client against the shape the backend actually returns.
///
/// These use a stubbed transport rather than a live server: the point is to
/// pin down the contract in `docs/API_CONTRACT.md`, not to check that the
/// network works. The payloads below are copied from real responses of
/// `uvicorn chessbot.api.app:app`, so if the backend changes shape one of
/// these fails.
///
/// The sign test is the one that matters. The backend sends `eval` from the
/// side-to-move's perspective; the UI wants White's. Get it backwards and
/// nothing crashes — the end-of-game chart just tells the story inverted.
library;

import 'dart:convert';

import 'package:chessbot_ui/engine/engine.dart';
import 'package:chessbot_ui/engine/http_engine.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';

const startPosition =
    'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
const blackToMove =
    'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1';

/// A client that answers every request with one canned response.
http.Client stubbing(int status, Object body) {
  return MockClient((_) async => http.Response(
        body is String ? body : jsonEncode(body),
        status,
        headers: {'content-type': 'application/json'},
      ));
}

void main() {
  group('a normal move', () {
    // Verbatim from the real backend at level medium.
    final payload = {
      'move': 'e2e4',
      'san': 'e4',
      'eval': 0.015111894905567169,
      'pv': ['e4', 'e6', 'd4', 'd5', 'Nc3', 'Nf6'],
      'ms': 590,
      'sims': 200,
      'game_over': false,
      'result': null,
    };

    test('is parsed field by field', () async {
      final engine = HttpEngine('http://x', client: stubbing(200, payload));
      final move = await engine.bestMove(startPosition, Difficulty.medium);

      expect(move.uci, 'e2e4');
      expect(move.thinkingMs, 590);
      expect(move.evaluation, closeTo(0.0151, 0.0001));
    });
  });

  group('the sign of eval', () {
    test('is left alone when White is to move', () async {
      final engine = HttpEngine(
        'http://x',
        client: stubbing(200, {
          'move': 'e2e4',
          'san': 'e4',
          'eval': 0.4,
          'pv': <String>[],
          'ms': 10,
          'sims': 50,
          'game_over': false,
          'result': null,
        }),
      );

      final move = await engine.bestMove(startPosition, Difficulty.easy);
      expect(move.evaluation, closeTo(0.4, 1e-9),
          reason: 'White to move: +0.4 for the mover is +0.4 for White');
    });

    test('is flipped when Black is to move', () async {
      final engine = HttpEngine(
        'http://x',
        client: stubbing(200, {
          'move': 'e7e5',
          'san': 'e5',
          'eval': 0.4,
          'pv': <String>[],
          'ms': 10,
          'sims': 50,
          'game_over': false,
          'result': null,
        }),
      );

      final move = await engine.bestMove(blackToMove, Difficulty.easy);
      expect(move.evaluation, closeTo(-0.4, 1e-9),
          reason: 'Black to move: +0.4 for the mover is -0.4 for White');
    });
  });

  group('errors become readable messages', () {
    test('a finished position is reported, not crashed on', () async {
      // What the backend really answers on a mated FEN.
      final engine = HttpEngine(
        'http://x',
        client: stubbing(200, {
          'move': null,
          'san': null,
          'eval': -1.0,
          'pv': <String>[],
          'ms': 0,
          'sims': 0,
          'game_over': true,
          'result': '0-1',
        }),
      );

      expect(
        () => engine.bestMove(startPosition, Difficulty.easy),
        throwsA(isA<EngineException>()),
      );
    });

    for (final (status, fragment) in [
      (422, 'bug in this app'),
      (429, 'Too many requests'),
      (503, 'busy'),
      (500, 'error (500)'),
    ]) {
      test('HTTP $status is explained in words', () async {
        final engine = HttpEngine(
          'http://x',
          client: stubbing(status, {'detail': 'whatever'}),
        );

        await expectLater(
          engine.bestMove(startPosition, Difficulty.easy),
          throwsA(
            isA<EngineException>().having(
              (e) => e.message,
              'message',
              contains(fragment),
            ),
          ),
        );
      });
    }

    test('an unreachable server does not leak a raw socket error', () async {
      final engine = HttpEngine(
        'http://x',
        client: MockClient((_) async => throw const SocketishFailure()),
      );

      await expectLater(
        engine.bestMove(startPosition, Difficulty.easy),
        throwsA(isA<EngineException>()),
      );
    });
  });

  group('health', () {
    test('true only once the model is actually loaded', () async {
      final up = HttpEngine(
        'http://x',
        client: stubbing(200, {
          'status': 'ok',
          'model_loaded': true,
          'device': 'cpu',
          'uptime_s': 21.0,
        }),
      );
      expect(await up.ping(), isTrue);

      // A Space still loading its weights answers 200 with model_loaded:false.
      // That is "waking", not "up".
      final waking = HttpEngine(
        'http://x',
        client: stubbing(200, {
          'status': 'ok',
          'model_loaded': false,
          'device': 'cpu',
          'uptime_s': 1.0,
        }),
      );
      expect(await waking.ping(), isFalse);
    });

    test('a dead server is false, not an exception', () async {
      final engine = HttpEngine(
        'http://x',
        client: MockClient((_) async => throw const SocketishFailure()),
      );
      expect(await engine.ping(), isFalse);
    });
  });

  group('the engine being busy', () {
    test('a 503 is retried once and then succeeds', () async {
      // The backend serialises searches: a 503 means someone else's move is
      // mid-flight, which on this page happens when the wake-up ping and a
      // quick first move overlap. Retrying turns a visible error into a
      // slightly slower move.
      var calls = 0;
      final engine = HttpEngine(
        'http://x',
        client: MockClient((_) async {
          calls++;
          if (calls == 1) {
            return http.Response(
              jsonEncode({'detail': 'search already running'}),
              503,
              headers: {'retry-after': '1', 'content-type': 'application/json'},
            );
          }
          return http.Response(
            jsonEncode({
              'move': 'e2e4',
              'san': 'e4',
              'eval': 0.1,
              'pv': <String>[],
              'ms': 500,
              'sims': 60,
              'game_over': false,
              'result': null,
            }),
            200,
            headers: {'content-type': 'application/json'},
          );
        }),
      );

      final move = await engine.bestMove(startPosition, Difficulty.medium);
      expect(move.uci, 'e2e4');
      expect(calls, 2, reason: 'exactly one retry, not a loop');
    });

    test('a second 503 gives up instead of retrying forever', () async {
      var calls = 0;
      final engine = HttpEngine(
        'http://x',
        client: MockClient((_) async {
          calls++;
          return http.Response(
            jsonEncode({'detail': 'busy'}),
            503,
            headers: {'retry-after': '1', 'content-type': 'application/json'},
          );
        }),
      );

      await expectLater(
        engine.bestMove(startPosition, Difficulty.medium),
        throwsA(isA<EngineException>()),
      );
      expect(calls, 2, reason: 'one attempt plus one retry, then it stops');
    });
  });
}

/// Stands in for whatever the platform throws when a host is unreachable.
class SocketishFailure implements Exception {
  const SocketishFailure();
}
