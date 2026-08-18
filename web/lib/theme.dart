/// The app's look: dark, with a muted brass accent.
///
/// Everything goes through `ThemeData`, so widgets carry no hardcoded colours
/// and restyling the app means touching only this file.
///
/// The palette: a near-black background that is warm rather than blue, surfaces
/// one step lighter to lift cards off it, and a single accent used sparingly.
/// One accent, reserved for what matters, is what separates a considered
/// interface from a christmas tree.
library;

import 'package:flutter/material.dart';

/// The accent: the primary button, the selected level, the chart line.
///
/// Muted brass rather than pure gold. `#FFD700` on a dark background reads as
/// electric yellow and looks like a game HUD; pulling the saturation down and
/// the lightness slightly back keeps the metal and loses the glare.
const Color accent = Color(0xFFC9A227);

/// A lighter brass, for text that sits on dark surfaces.
///
/// [accent] itself is too dim against [surface] for small text — this one
/// clears the contrast bar for body copy.
const Color accentSoft = Color(0xFFE0BF57);

/// The page background. Warm-neutral, not blue-black.
const Color background = Color(0xFF14130F);

/// Cards and panels, one step above the background.
const Color surface = Color(0xFF1E1C17);

/// Borders: barely there, meant to delimit rather than draw.
const Color outline = Color(0xFF32302A);

/// Primary text.
const Color textPrimary = Color(0xFFF0EDE6);

/// Secondary text: descriptions, captions.
const Color textSecondary = Color(0xFFB0AAA0);

/// Muted text: hints, technical labels.
const Color textMuted = Color(0xFF7C766C);

/// Errors and losses. Desaturated so it sits with the rest of the palette.
const Color danger = Color(0xFFC5544D);

/// The style for technical labels: engine strength, level name.
///
/// The monospace is deliberate — it gives the interface the air of an
/// instrument rather than a generic app — but it belongs on short labels, not
/// on running text.
const TextStyle monoLabel = TextStyle(
  fontFamily: 'monospace',
  fontFamilyFallback: ['Courier New', 'monospace'],
  fontSize: 12.5,
  letterSpacing: 0.3,
  color: textMuted,
);

/// The two board shades.
///
/// Chosen directly, not derived. The previous library took a single colour and
/// lightened it by a fixed amount in HSL, which forced the light squares
/// bright whatever base it was given — a beige board on a dark page. Drawing
/// the board here means picking both tones outright.
///
/// They sit close in hue to the surfaces around them and far enough apart in
/// lightness to read as a chessboard at a glance.
const Color boardLight = Color(0xFF6E6A62);
const Color boardDark = Color(0xFF3A3733);

/// The pieces.
///
/// Warm off-white and near-black rather than pure #FFF/#000: full white glares
/// against a dark board, and full black loses its outline on the dark squares.
/// Each glyph carries a thin shadow in the opposite colour, so both stay
/// legible on both shades.
const Color pieceWhite = Color(0xFFF2EFE9);
const Color pieceBlack = Color(0xFF15130F);

ThemeData buildDarkTheme() {
  final scheme =
      ColorScheme.fromSeed(
        seedColor: accent,
        brightness: Brightness.dark,
      ).copyWith(
        surface: background,
        primary: accent,
        // Text on the accent must be dark: brass is a light colour, and white on
        // top of it is unreadable.
        onPrimary: const Color(0xFF1A1405),
        outline: outline,
      );

  return ThemeData(
    useMaterial3: true,
    colorScheme: scheme,
    scaffoldBackgroundColor: background,
    // No global `fontFamily`: `monospace` is not a bundled font but a generic
    // name the browser resolves on its own (Consolas on Windows, Menlo on
    // macOS), and it reads worse for running text. It survives only where it
    // earns its place — see `monoLabel`.
    textTheme: const TextTheme(
      headlineMedium: TextStyle(
        fontWeight: FontWeight.w600,
        letterSpacing: -0.5,
        color: textPrimary,
      ),
      headlineSmall: TextStyle(fontWeight: FontWeight.w600, color: textPrimary),
      titleMedium: TextStyle(fontWeight: FontWeight.w600, color: textPrimary),
      bodyMedium: TextStyle(color: textSecondary),
      bodySmall: TextStyle(color: textMuted),
    ),
    filledButtonTheme: FilledButtonThemeData(
      style: FilledButton.styleFrom(
        backgroundColor: accent,
        foregroundColor: const Color(0xFF1A1405),
        textStyle: const TextStyle(
          fontWeight: FontWeight.w600,
          letterSpacing: 0.3,
        ),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
      ),
    ),
    textButtonTheme: TextButtonThemeData(
      style: TextButton.styleFrom(foregroundColor: textMuted),
    ),
    cardTheme: CardThemeData(
      color: surface,
      elevation: 0,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12),
        side: const BorderSide(color: outline),
      ),
    ),
    progressIndicatorTheme: const ProgressIndicatorThemeData(color: accent),
  );
}
