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
BATCH_SIZE      = 64
LR              = 5e-4
WEIGHT_DECAY    = 1e-4    # L2 regularisation
NUM_EPOCHS      = 200
EARLY_STOP      = 30      # stop if val F-measure doesn't improve for N epochs
BEAT_POS_WEIGHT = 3.0     # up-weight beat frames 2.5降低到3.0
TEMPO_LOSS_W    = 0.3     # disabled: tempo task dilutes beat gradients on diverse datasets  改成0.3
GRAD_CLIP       = 1.0
DOWNBEAT_LOSS_W = 0.3
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

    for mel, beat, downbeat, tempo in loader:
        mel, beat, downbeat, tempo = mel.to(device), beat.to(device), downbeat.to(device), tempo.to(device)

        beat_pred, downbeat_pred, tempo_logits = model(mel)

        loss = (weighted_bce(beat_pred, beat)
                + DOWNBEAT_LOSS_W * weighted_bce(downbeat_pred, downbeat)
                + TEMPO_LOSS_W * F.binary_cross_entropy(
                    torch.sigmoid(tempo_logits), tempo.float()))
        total_loss += loss.item()

        for pred_np, true_np in zip(beat_pred.cpu().numpy(),
                                     beat.cpu().numpy()):
            fmeasures.append(beat_fmeasure(pred_np, true_np))

    return total_loss / len(loader), float(np.mean(fmeasures))
@torch.no_grad()
def evaluate_with_search(model, loader, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_trues = [], []

    for mel, beat, downbeat, tempo in loader:
        mel, beat, downbeat, tempo = mel.to(device), beat.to(device), downbeat.to(device), tempo.to(device)
        beat_pred, downbeat_pred, tempo_logits = model(mel)
        loss = (weighted_bce(beat_pred, beat)
                + DOWNBEAT_LOSS_W * weighted_bce(downbeat_pred, downbeat)
                + TEMPO_LOSS_W * F.binary_cross_entropy(
                    torch.sigmoid(tempo_logits), tempo.float()))
        total_loss += loss.item()
        all_preds.append(beat_pred.cpu().numpy())
        all_trues.append(beat.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_trues = np.concatenate(all_trues, axis=0)

    # grid search
    best_f, best_thresh, best_dist = 0.0, PEAK_THRESHOLD, PEAK_MIN_DIST
    for thresh in [0.3, 0.4, 0.5, 0.6]:
        for min_d in [3, 4, 5, 6, 7]:
            
            # 暫時覆蓋 peak_pick 預設參數
            fscores = []
            for n in range(len(all_preds)):
                pred_frames = peak_pick(all_preds[n], threshold=thresh, min_dist=min_d)
                true_frames = np.where(all_trues[n] > 0.9)[0]
                if len(pred_frames) == 0 and len(true_frames) == 0:
                    fscores.append(1.0)
                    continue
                if len(pred_frames) == 0 or len(true_frames) == 0:
                    fscores.append(0.0)
                    continue
                pred_sec = pred_frames * HOP_LENGTH / SAMPLE_RATE
                true_sec = true_frames * HOP_LENGTH / SAMPLE_RATE
                matched, tp = set(), 0
                for pt in pred_sec:
                    dists = np.abs(true_sec - pt)
                    j = int(np.argmin(dists))
                    if dists[j] <= TOLERANCE_SEC and j not in matched:
                        tp += 1
                        matched.add(j)
                p = tp / len(pred_frames)
                r = tp / len(true_frames)
                fscores.append(2*p*r/(p+r) if (p+r) > 0 else 0.0)

            mean_f = float(np.mean(fscores))
            if mean_f > best_f:
                best_f, best_thresh, best_dist = mean_f, thresh, min_d

    return total_loss / len(loader), best_f, best_thresh, best_dist
# ── Training ─────────────────────────────────────────────────────────────────

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}\n")

    # ── Datasets ─────────────────────────────────────────────────────────────
    DATA_ROOT = r'C:/Users/ricky/Documents/python_learning/Final_Project/dataset'
    cache = f'{DATA_ROOT}/mel_cache'
    train_ds = BallroomDataset(DATA_ROOT, split='train', cache_dir=cache)
    val_ds   = BallroomDataset(DATA_ROOT, split='val',   cache_dir=cache)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2,
                              pin_memory=True)

    # ── Model ────────────────────────────────────────────────────────────────
    model = BeatTCN().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters : {n_params:,}\n")

    # ── Optimiser & scheduler ────────────────────────────────────────────────
    optimiser = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=200, eta_min=1e-6)

    # ── Resume from checkpoint if available ──────────────────────────────────
    ckpt_dir   = Path('checkpoints')
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_file = ckpt_dir / 'latest_model_new_arch.pt'
    best_val_f = 0.0
    start_epoch = 1
    no_improve = 0

    if ckpt_file.exists():
        ckpt = torch.load(ckpt_file, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        start_epoch = ckpt['epoch'] + 1
        no_improve  = ckpt.get('no_improve', 0)

    # best_val_f 從 best checkpoint 讀，不從 latest 讀
    best_ckpt_file = ckpt_dir / 'best_model_new_arch.pt'
    if best_ckpt_file.exists():
        best_ckpt  = torch.load(best_ckpt_file, map_location=device)
        best_val_f = best_ckpt['val_f']
    else:
        best_val_f = 0.0

    print(f"Resumed from epoch {ckpt['epoch']} "
          f"(best val_F={best_val_f:.4f}, continuing from epoch {start_epoch}, "
          f"no_improve={no_improve})\n")

    print(f"{'Epoch':>5}  {'train_loss':>10}  {'val_loss':>8}  {'val_F':>6}  {'lr':>8}  {'time':>5}")
    print("-" * 60)

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        # warmup：前5個epoch讓lr從小慢慢爬上來
        if epoch <= 5:
            for g in optimiser.param_groups:
                g['lr'] = LR * epoch / 5
        # 動態調整 tempo loss weight：早期多學 tempo，後期專注 beat
        TEMPO_LOSS_W = max(0.1, 0.3 * (1 - epoch / NUM_EPOCHS))
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for mel, beat, downbeat, tempo in train_loader:
            mel, beat, downbeat, tempo = mel.to(device), beat.to(device), downbeat.to(device), tempo.to(device)
            beat_pred, downbeat_pred, tempo_logits = model(mel)
            loss = (weighted_bce(beat_pred, beat)
                        + DOWNBEAT_LOSS_W * weighted_bce(downbeat_pred, downbeat)
                        + TEMPO_LOSS_W * F.binary_cross_entropy(
                            torch.sigmoid(tempo_logits), tempo.float()))

            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimiser.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        ##每5個epoch做一次grid search
        if epoch % 5 == 0:
            val_loss, val_f, best_thresh, best_dist = evaluate_with_search(model, val_loader, device)
        else:
            val_loss, val_f = evaluate(model, val_loader, device)
            best_thresh, best_dist = PEAK_THRESHOLD, PEAK_MIN_DIST
        if epoch > 5:   # warmup 結束後才讓 scheduler 接手
            scheduler.step()

        lr_now = optimiser.param_groups[0]['lr']
        print(f"{epoch:5d}  {train_loss:10.4f}  {val_loss:8.4f}  "
            f"{val_f:6.4f}  {lr_now:8.6f}  {time.time()-t0:4.0f}s  "
            f"[t={best_thresh} d={best_dist}]")

        if val_f > best_val_f:
            best_val_f = val_f
            no_improve = 0
            torch.save({
                'epoch'      : epoch,
                'model_state': model.state_dict(),
                'val_f'      : val_f,
                'peak_threshold': best_thresh,
                'peak_min_dist' : best_dist,
            }, ckpt_dir / 'best_model_new_arch.pt')
            print(f"  → best model saved  (val_F={val_f:.4f}, "
                  f"threshold={best_thresh}, min_dist={best_dist})")
        # 每個 epoch 都存 latest
        
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP:
                print(f"\nEarly stop at epoch {epoch} "
                      f"(no improvement for {EARLY_STOP} epochs)")
                torch.save({
                    'epoch'          : epoch,
                    'model_state'    : model.state_dict(),
                    'val_f'          : val_f,
                    'peak_threshold' : best_thresh,
                    'peak_min_dist'  : best_dist,
                }, ckpt_dir / 'latest_model_new_arch.pt')
                break
        torch.save({
            'epoch'          : epoch,
            'model_state'    : model.state_dict(),
            'val_f'          : val_f,
            'peak_threshold' : best_thresh,
            'peak_min_dist'  : best_dist,
        }, ckpt_dir / 'latest_model_new_arch.pt')

    print(f"\nDone. Best val F-measure : {best_val_f:.4f}")
    print(f"Checkpoint : {ckpt_dir / 'best_model_new_arch.pt'}")


if __name__ == '__main__':
    train()