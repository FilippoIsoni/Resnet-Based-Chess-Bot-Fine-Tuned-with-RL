/// App configuration, fixed at build time.
library;

/// URL of the backend running the engine.
///
/// Set at build time, not in code:
///
///     flutter run -d chrome                                     # local backend
///     flutter build web --dart-define=BACKEND_URL=https://...   # deployed
///
/// It must be `const`: `String.fromEnvironment` is substituted by the
/// compiler, and in a plain variable a release build would keep the default.
///
/// The default points at a backend running on this machine, which is what
/// `uvicorn chessbot.api.app:app` serves. **A release build must always pass
/// `--dart-define`**: a page served over HTTPS cannot call `http://127.0.0.1`,
/// and the browser blocks it as mixed content with no visible error — the app
/// simply looks broken. The Pages workflow passes it; see
/// `.github/workflows/pages.yml`.
const String backendUrl = String.fromEnvironment(
  'BACKEND_URL',
  defaultValue: 'http://127.0.0.1:8000',
);
