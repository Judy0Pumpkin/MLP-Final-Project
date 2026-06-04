import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from constants import SAMPLE_RATE, HOP_LENGTH
from dataset   import BallroomDataset
from model     import BeatTCN

# ── Hyperparameters ──────────────────────────────────────────────────────────
BATCH_SIZE      = 32
LR              = 1e-3
WEIGHT_DECAY    = 1e-4    # L2 regularisation
NUM_EPOCHS      = 100
EARLY_STOP      = 15      # stop if val F-measure doesn't improve for N epochs
BEAT_POS_WEIGHT = 5.0     # up-weight beat frames
TEMPO_LOSS_W    = 0.0     # disabled: tempo task dilutes beat gradients on diverse datasets
GRAD_CLIP       = 1.0

# ── Beat F-measure settings ───────────────────────────────────────────────────
TOLERANCE_SEC  = 0.07     # 70 ms standard window for beat evaluation
PEAK_THRESHOLD = 0.4
PEAK_MIN_DIST  = 5        # min frames between detected beats


# ── Helpers ──────────────────────────────────────────────────────────────────

def weighted_bce(pred, target, pos_weight=BEAT_POS_WEIGHT):
    """BCE with higher weight on positive (beat) frames."""
    w = torch.where(target > 0.5,
                    torch.full_like(target, pos_weight),
                    torch.ones_like(target))
    return F.binary_cross_entropy(pred, target, weight=w)


def peak_pick(act, threshold=PEAK_THRESHOLD, min_dist=PEAK_MIN_DIST):
    """Return frame indices of local maxima above threshold."""
    peaks = []
    for i in range(1, len(act) - 1):
        if act[i] > threshold and act[i] >= act[i-1] and act[i] >= act[i+1]:
            if not peaks or i - peaks[-1] >= min_dist:
                peaks.append(i)
    return np.array(peaks, dtype=np.int32)


def beat_fmeasure(pred_act, true_act, tol=TOLERANCE_SEC):
    """F-measure between predicted and ground-truth beat activations."""
    pred_frames = peak_pick(pred_act)
    # threshold=0.9 取 Gaussian 峰值，避免 smoothing 造成每個 beat 被算成多個 GT frame
    true_frames = np.where(true_act > 0.9)[0]

    if len(pred_frames) == 0 and len(true_frames) == 0:
        return 1.0
    if len(pred_frames) == 0 or len(true_frames) == 0:
        return 0.0

    pred_sec = pred_frames * HOP_LENGTH / SAMPLE_RATE
    true_sec = true_frames * HOP_LENGTH / SAMPLE_RATE

    matched_true = set()
    tp = 0
    for pt in pred_sec:
        dists = np.abs(true_sec - pt)
        j = int(np.argmin(dists))
        if dists[j] <= tol and j not in matched_true:
            tp += 1
            matched_true.add(j)

    precision = tp / len(pred_frames)
    recall    = tp / len(true_frames)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    fmeasures  = []

    for mel, beat, tempo in loader:
        mel, beat, tempo = mel.to(device), beat.to(device), tempo.to(device)

        beat_pred, tempo_logits = model(mel)

        loss = (weighted_bce(beat_pred, beat)
                + TEMPO_LOSS_W * F.cross_entropy(tempo_logits, tempo))
        total_loss += loss.item()

        # Per-sample F-measure (CPU numpy)
        for pred_np, true_np in zip(beat_pred.cpu().numpy(),
                                     beat.cpu().numpy()):
            fmeasures.append(beat_fmeasure(pred_np, true_np))

    return total_loss / len(loader), float(np.mean(fmeasures))


# ── Training ─────────────────────────────────────────────────────────────────

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}\n")

    # ── Datasets ─────────────────────────────────────────────────────────────
    cache = 'dataset/mel_cache'
    train_ds = BallroomDataset('dataset', split='train', cache_dir=cache)
    val_ds   = BallroomDataset('dataset', split='val',   cache_dir=cache)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0,
                              pin_memory=(device.type == 'cuda'))
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0,
                              pin_memory=(device.type == 'cuda'))

    # ── Model ────────────────────────────────────────────────────────────────
    model = BeatTCN().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters : {n_params:,}\n")

    # ── Optimiser & scheduler ────────────────────────────────────────────────
    optimiser = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode='max', factor=0.5, patience=5
    )

    # ── Resume from checkpoint if available ──────────────────────────────────
    ckpt_dir   = Path('checkpoints')
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_file  = ckpt_dir / 'best_model.pt'
    best_val_f = 0.0
    start_epoch = 1
    no_improve = 0

    if ckpt_file.exists():
        ckpt = torch.load(ckpt_file, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        best_val_f  = ckpt['val_f']       # 恢復基準
        start_epoch = ckpt['epoch'] + 1   # 從下一個 epoch 繼續
        print(f"Loaded weights from epoch {ckpt['epoch']} "
              f"(old val_F={ckpt['val_f']:.4f}, restarting counters)\n")

    print(f"{'Epoch':>5}  {'train_loss':>10}  {'val_loss':>8}  {'val_F':>6}  {'lr':>8}  {'time':>5}")
    print("-" * 60)

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for mel, beat, tempo in train_loader:
            mel, beat, tempo = mel.to(device), beat.to(device), tempo.to(device)

            beat_pred, tempo_logits = model(mel)

            loss = (weighted_bce(beat_pred, beat)
                    + TEMPO_LOSS_W * F.cross_entropy(tempo_logits, tempo))

            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimiser.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss, val_f = evaluate(model, val_loader, device)
        scheduler.step(val_f)

        lr_now = optimiser.param_groups[0]['lr']
        print(f"{epoch:5d}  {train_loss:10.4f}  {val_loss:8.4f}  "
              f"{val_f:6.4f}  {lr_now:8.6f}  {time.time()-t0:4.0f}s")

        if val_f > best_val_f:
            best_val_f = val_f
            no_improve = 0
            torch.save({
                'epoch'      : epoch,
                'model_state': model.state_dict(),
                'val_f'      : val_f,
            }, ckpt_dir / 'best_model.pt')
            print(f"  → best model saved  (val_F={val_f:.4f})")
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP:
                print(f"\nEarly stop at epoch {epoch} "
                      f"(no improvement for {EARLY_STOP} epochs)")
                break

    print(f"\nDone. Best val F-measure : {best_val_f:.4f}")
    print(f"Checkpoint : {ckpt_dir / 'best_model.pt'}")


if __name__ == '__main__':
    train()
