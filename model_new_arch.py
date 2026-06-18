import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import N_MELS
from dataset   import N_TEMPO_BINS


# ── Building blocks ──────────────────────────────────────────────────────────

class CausalConv1d(nn.Module):
    """Conv1d padded only on the left → no lookahead, safe for real-time."""

    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        super().__init__()
        self.pad  = (kernel_size - 1) * dilation   # left-only padding
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size,
                              dilation=dilation, padding=0)

    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0)))


class TCNBlock(nn.Module):
    """Two causal dilated conv1d + residual skip connection."""

    def __init__(self, n_ch, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        self.conv1 = CausalConv1d(n_ch, n_ch, kernel_size, dilation)
        self.conv2 = CausalConv1d(n_ch, n_ch, kernel_size, dilation)
        self.bn1   = nn.BatchNorm1d(n_ch)
        self.bn2   = nn.BatchNorm1d(n_ch)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        out = F.elu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        return F.elu(out + x)          # residual


# ── Main model ───────────────────────────────────────────────────────────────

class BeatTCN(nn.Module):
    """
    CNN + Causal-TCN beat tracker.

    CNN  : compresses the mel frequency axis (128 → 1), keeping the time axis
    TCN  : stacked dilated causal conv1d blocks to capture long-range rhythm
    Heads: beat activation per frame + global tempo classification

    Input  : (B, 1, N_MELS, T)
    Output : beat_act     (B, T)            – sigmoid, frame-level probability
             tempo_logits (B, N_TEMPO_BINS) – raw logits for CrossEntropyLoss
    """

    # TCN hyper-parameters
    TCN_CHANNELS  = 64
    TCN_KERNEL    = 3
    TCN_DILATIONS = [1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32]   # receptive field ≈ 4*(1+2+…+32)*2 = 252 frames

    def __init__(self, n_mels=N_MELS, n_tempo_bins=N_TEMPO_BINS, dropout=0.1):
        super().__init__()
        assert n_mels == 128, "CNN is designed for N_MELS=128"

        # ── CNN ───────────────────────────────────
        self.cnn = nn.Sequential(
            # block 1
            nn.Conv2d(1,  16, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(16), nn.ELU(),
            nn.AvgPool2d((2, 1)),                   # freq: 128 → 64

            # block 2
            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32), nn.ELU(),
            nn.AvgPool2d((2, 1)),                   # freq: 64 → 32

            # block 3
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64), nn.ELU(),
            nn.AvgPool2d((4, 1)),                   # freq: 32 → 8

            # compress remaining 8 freq bins → 1
            nn.Conv2d(64, self.TCN_CHANNELS, kernel_size=(8, 1)),
            nn.BatchNorm2d(self.TCN_CHANNELS), nn.ELU(),
            # output: (B, 64, 1, T)
        )

        # ── TCN ─────────────────────────────────────────
        self.tcn_blocks = nn.ModuleList([
            TCNBlock(self.TCN_CHANNELS, self.TCN_KERNEL, dilation=d, dropout=dropout)
            for d in self.TCN_DILATIONS])

        # ── Beat head: frame-level activation ────────────────────────────────
        self.beat_head = nn.Conv1d(self.TCN_CHANNELS, 1, kernel_size=1)
        self.downbeat_head = nn.Conv1d(self.TCN_CHANNELS, 1, kernel_size=1)
        # ── Tempo head: global BPM classification ────────────────────────────
        self.tempo_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),      # (B, 64, 1)
            nn.Flatten(),                 # (B, 64)
            nn.Linear(self.TCN_CHANNELS, n_tempo_bins),
        )

    def forward(self, x):
        feat = self.cnn(x).squeeze(2)   # (B, 64, T)
    
        skip_sum = torch.zeros_like(feat)
        for block in self.tcn_blocks:
            feat = block(feat)
            skip_sum = skip_sum + feat  # 累加每層輸出（skip aggregation）
    
        beat_act      = torch.sigmoid(self.beat_head(feat)).squeeze(1)
        downbeat_act  = torch.sigmoid(self.downbeat_head(feat)).squeeze(1)
        tempo_logits  = self.tempo_head(skip_sum)
        return beat_act, downbeat_act, tempo_logits


# ── Smoke-test ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    from constants import FIXED_FRAMES

    B, T = 4, FIXED_FRAMES
    x    = torch.randn(B, 1, N_MELS, T)
    model = BeatTCN()

    beat, tempo = model(x)
    print(f"Input        : {tuple(x.shape)}")
    print(f"beat_act     : {tuple(beat.shape)}   (expected ({B}, {T}))")
    print(f"tempo_logits : {tuple(tempo.shape)}  (expected ({B}, {N_TEMPO_BINS}))")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters   : {n_params:,}")
