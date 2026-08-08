"""Le 20 posizioni di riferimento del Gate 1 (§2.5).

Sono scelte a mano, non estratte a caso: ognuna copre un caso che l'encoder puo
sbagliare in modo silenzioso. Vivono qui e non nel JSON perche il JSON e l'OUTPUT
atteso — rigenerabile — mentre l'INPUT deve restare fisso e leggibile.

Rigenerare il golden con:

    python scripts/gen_golden.py
"""

from __future__ import annotations

# (nome, FEN, mossa UCI da codificare come etichetta)
GOLDEN_POSITIONS: tuple[tuple[str, str, str], ...] = (
    (
        "posizione_iniziale",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "e2e4",
    ),
    (
        "dopo_e4_tocca_al_nero",
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "e7e5",
    ),
    (
        "kiwipete",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "e1g1",
    ),
    (
        "kiwipete_specchiata",
        "r3k2r/pppbbppp/2n2q1P/1P2p3/3pn3/BN2PNP1/P1PPQPB1/R3K2R b KQkq - 0 1",
        "e8g8",
    ),
    (
        "perft_position_3",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "e2e4",
    ),
    (
        # Posizione bloccatissima: solo 6 mosse legali, il pedone in a7 non puo
        # promuovere perche a8 e occupata. Utile proprio perche e un caso stretto.
        "perft_position_4",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "d2d4",
    ),
    (
        "perft_position_5_promozione_con_cattura",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "d7c8q",
    ),
    (
        "en_passant_disponibile_bianco",
        "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
        "e5f6",
    ),
    (
        "en_passant_disponibile_nero",
        "rnbqkbnr/pppp1ppp/8/8/3pP3/5N2/PPP2PPP/RNBQKB1R b KQkq e3 0 3",
        "d4e3",
    ),
    (
        "nessun_arrocco",
        "r3k2r/8/8/8/8/8/8/R3K2R w - - 0 1",
        "e1e2",
    ),
    (
        "solo_arrocco_corto_bianco",
        "r3k2r/8/8/8/8/8/8/R3K2R w K - 0 1",
        "e1g1",
    ),
    (
        "promozione_donna_bianco",
        "8/P6k/8/8/8/8/7K/8 w - - 0 1",
        "a7a8q",
    ),
    (
        "sottopromozione_cavallo_bianco",
        "8/P6k/8/8/8/8/7K/8 w - - 0 1",
        "a7a8n",
    ),
    (
        "promozione_donna_nero",
        "8/7K/8/8/8/8/p6k/8 b - - 0 1",
        "a2a1q",
    ),
    (
        "sottopromozione_torre_nero",
        "8/7K/8/8/8/8/p6k/8 b - - 0 1",
        "a2a1r",
    ),
    (
        "promozione_con_cattura",
        "1n5k/P7/8/8/8/8/7K/8 w - - 0 1",
        "a7b8q",
    ),
    (
        "halfmove_clock_alto",
        "8/8/4k3/8/8/4K3/8/8 w - - 87 120",
        "e3e4",
    ),
    (
        "finale_re_e_pedone_nero_muove",
        "8/8/8/4k3/8/8/4P3/4K3 b - - 5 40",
        "e5e4",
    ),
    (
        "matto_del_barbiere",
        "rnbqkbnr/pppp1ppp/8/4p3/5PP1/8/PPPPP2P/RNBQKBNR b KQkq g3 0 2",
        "d8h4",
    ),
    (
        "mossa_di_cavallo_centrale",
        "8/8/3k4/8/4N3/8/3K4/8 w - - 12 60",
        "e4f6",
    ),
)
