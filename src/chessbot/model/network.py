"""ResNet con due teste: policy e value WDL (§3.1 del piano).

    Input 19x8x8
      Conv 3x3, 128 filtri, BatchNorm, ReLU
      8 x blocco residuo: Conv3x3-BN-ReLU-Conv3x3-BN-(+skip)-ReLU
        policy head:  Conv 1x1 (32 canali) -> flatten -> Linear -> 4672 logit
        value head:   Conv 1x1 (8 canali)  -> Linear(128) -> Linear(3) -> WDL

**12,0M parametri**, non i ~3,5M stimati dal piano. La stima non torna con l'architettura
che il piano stesso specifica: il solo `Linear(32*8*8 -> 4672)` della policy head vale
9,57M parametri, e da solo supera il totale dichiarato.

    stem              22.144   0,2%
    8 blocchi      2.363.392  19,6%
    policy head    9.577.088  79,6%   <- il Linear finale
    value head        67.091   0,6%
    TOTALE        12.029.715

L'architettura resta quella specificata: in fp32 sono 48 MB di pesi, e con batch 512 il
training sta comodamente nei 6 GB della 3050. Se un domani servisse comprimere, la strada
standard e la policy head "convoluzionale pura" di AlphaZero — Conv 1x1 da 128 a 73
canali, il cui output 73x8x8 = 4672 e gia lo spazio delle mosse, senza alcun Linear.
Costerebbe 9.344 parametri invece di 9,57M, portando la rete a ~2,5M. Vedi
docs/DECISIONS.md.

## Perche le convoluzioni

Le relazioni scacchistiche sono **locali e spaziali**: catene di pedoni, case deboli,
difese reciproche. Una conv 3x3 guarda una casa e le otto vicine, esattamente la scala
giusta. Impilandone 8+1 il campo recettivo copre l'intera scacchiera, quindi la rete puo
comunque cogliere relazioni fra angoli opposti.

## Perche i blocchi residui

Senza le connessioni skip, una rete profonda fatica ad allenarsi: il gradiente si
attenua passando all'indietro per molti strati. Il blocco residuo calcola `x + f(x)`
invece di `f(x)`, cosi il gradiente ha sempre una strada diretta per tornare indietro.
E il motivo per cui si possono impilare 8 blocchi senza che il training si blocchi.

## Perche la testa WDL a 3 uscite e non un tanh scalare

Con uno scalare, una posizione **realmente pari** e una **sbilanciata ma dall'esito
incerto** valgono entrambe 0: informazione persa. Con tre probabilita (sconfitta, patta,
vittoria) le due situazioni sono distinguibili, e la cross-entropy su tre classi da
gradienti piu informativi della MSE su uno scalare.

Conta soprattutto qui, dove la value head e la componente debole del progetto: avendo
escluso la distillazione da Stockfish, ogni bit di segnale in piu e prezioso
(criticita #5).

> **Entrambe le teste servono da subito.** Senza value head l'MCTS non puo esistere: e
> lei che valuta le foglie dell'albero. Allenarla ora evita di rifare tutto fra un mese.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from chessbot.encoding import BOARD_SIZE, N_MOVES, N_PLANES

# Le tre classi della testa WDL, nell'ordine usato dallo storage (`data/pgn.py`).
WDL_CLASSES = 3


@dataclass
class NetworkConfig:
    """Iperparametri dell'architettura, da `configs/default.yaml` sezione `model`."""

    channels: int = 128
    blocks: int = 8
    policy_head_channels: int = 32
    value_head_channels: int = 8
    value_hidden: int = 128

    def __post_init__(self) -> None:
        if self.blocks < 1:
            raise ValueError(f"servono almeno 1 blocco, ricevuti {self.blocks}")
        if self.channels < 1:
            raise ValueError(f"canali non validi: {self.channels}")


class ResidualBlock(nn.Module):
    """Conv3x3 - BN - ReLU - Conv3x3 - BN - (+ input) - ReLU.

    Il `bias=False` nelle convoluzioni non e una svista: la BatchNorm che segue sottrae
    la media, quindi qualunque bias verrebbe cancellato subito dopo. Tenerlo sarebbe
    parametri sprecati.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # La somma avviene PRIMA della ReLU finale: e la variante originale di He et al.,
        # e vuol dire che il ramo skip passa lineare, senza essere schiacciato a zero.
        return torch.relu(out + identity)


class PolicyHead(nn.Module):
    """Dalla rappresentazione condivisa ai 4672 logit, uno per mossa possibile.

    La conv 1x1 comprime 128 canali a 32 senza guardare i vicini: serve solo a ridurre
    la dimensione prima del Linear, che altrimenti avrebbe 128*64*4672 = 38M parametri —
    piu di dieci volte l'intera rete.
    """

    def __init__(self, channels: int, head_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, head_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(head_channels)
        self.fc = nn.Linear(head_channels * BOARD_SIZE * BOARD_SIZE, N_MOVES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn(self.conv(x)))
        return self.fc(out.flatten(start_dim=1))


class ValueHead(nn.Module):
    """Dalla rappresentazione condivisa a tre logit: sconfitta, patta, vittoria.

    Restituisce **logit**, non probabilita: la softmax la applica la cross-entropy, che
    lo fa in modo numericamente stabile. Applicarla qui e poi passare le probabilita
    alla loss e un errore classico che produce gradienti sbagliati.
    """

    def __init__(self, channels: int, head_channels: int, hidden: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, head_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(head_channels)
        self.fc1 = nn.Linear(head_channels * BOARD_SIZE * BOARD_SIZE, hidden)
        self.fc2 = nn.Linear(hidden, WDL_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn(self.conv(x)))
        out = torch.relu(self.fc1(out.flatten(start_dim=1)))
        return self.fc2(out)


class ChessNet(nn.Module):
    """La rete completa: tronco condiviso, due teste.

        logit_policy, logit_wdl = net(planes)

    Il tronco e condiviso di proposito: "capire la posizione" e lo stesso lavoro sia per
    scegliere la mossa sia per stimare chi sta meglio. Due reti separate imparerebbero
    due volte le stesse cose, con il doppio dei parametri.
    """

    def __init__(self, config: NetworkConfig | None = None) -> None:
        super().__init__()
        self.config = config or NetworkConfig()
        c = self.config.channels

        self.stem = nn.Sequential(
            nn.Conv2d(N_PLANES, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*(ResidualBlock(c) for _ in range(self.config.blocks)))
        self.policy_head = PolicyHead(c, self.config.policy_head_channels)
        self.value_head = ValueHead(c, self.config.value_head_channels, self.config.value_hidden)

    def forward(self, planes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """planes (B, 19, 8, 8) -> (logit policy (B, 4672), logit WDL (B, 3))."""
        if planes.dim() != 4 or planes.shape[1:] != (N_PLANES, BOARD_SIZE, BOARD_SIZE):
            raise ValueError(
                f"attesa forma (B, {N_PLANES}, {BOARD_SIZE}, {BOARD_SIZE}), "
                f"ricevuta {tuple(planes.shape)}"
            )
        features = self.blocks(self.stem(planes))
        return self.policy_head(features), self.value_head(features)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def describe(self) -> str:
        n = self.count_parameters()
        return (
            f"ChessNet: {self.config.blocks} blocchi x {self.config.channels} canali, "
            f"{n:,} parametri ({n * 4 / 1e6:.1f} MB in fp32)"
        )


def masked_policy_logits(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """Azzera la probabilita delle mosse illegali, mettendo i loro logit a -inf.

    > **Criticita #12 — questa funzione va usata IDENTICA in training e in inferenza.**
    > Se in training la rete distribuisce probabilita anche sulle mosse illegali e in
    > inferenza no (o viceversa), sta ottimizzando una distribuzione diversa da quella
    > che poi userera. Non crasha: la forza semplicemente non arriva dove dovrebbe.
    >
    > Per questo esiste una funzione sola, esportata, invece di due righe copiate nei
    > due punti.

    > **La sentinella dipende dal dtype.** In fp32 si usa `-1e9`, ma in fp16 il massimo
    > rappresentabile e 65504: `-1e9` va in overflow e PyTorch solleva
    > `RuntimeError: value cannot be converted to type c10::Half without overflow`.
    > Con la precisione mista attiva (che e il default sulla 3050) i logit arrivano qui
    > in fp16, quindi serve una soglia adatta al tipo.
    >
    > Intercettato dallo smoke test del training, non dai test unitari: quelli girano in
    > fp32, dove il bug non si manifesta.

    Non si usa `-inf` letterale perche produce NaN appena entra in un prodotto con zero,
    mentre un numero finito molto negativo dopo la softmax da comunque zero esatto.
    """
    if logits.shape != legal_mask.shape:
        raise ValueError(
            f"logit {tuple(logits.shape)} e maschera {tuple(legal_mask.shape)} non combaciano"
        )
    # `finfo.min` e il piu piccolo rappresentabile nel dtype: /2 lascia margine perche
    # eventuali somme a valle non vadano in overflow.
    sentinel = torch.finfo(logits.dtype).min / 2
    return logits.masked_fill(~legal_mask, sentinel)


__all__ = [
    "WDL_CLASSES",
    "ChessNet",
    "NetworkConfig",
    "PolicyHead",
    "ResidualBlock",
    "ValueHead",
    "masked_policy_logits",
]
