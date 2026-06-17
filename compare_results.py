"""
compare_results.py

對 test split 的每首歌，分別用三個模型評估：
  1. Librosa baseline        (librosa.beat.beat_track，全曲)
  2. BeatTCN v1              (best_model_F=0.6457.pt，model.py 架構)
  3. BeatTCN v2 (new arch)   (best_model_new_arch.pt，model_new_arch.py 架構)

計算 Beat F-measure (tolerance=70ms)，印出統計摘要、存 CSV、畫 per-dataset bar chart。

執行：
    python compare_results.py
"""

import os
os.environ['NUMBA_DISABLE_SVML'] = '1'

import csv
from pathlib import Path

import numpy as np
import librosa
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from constants import SAMPLE_RATE, HOP_LENGTH, N_MELS, N_FFT, F_MIN, F_MAX, FIXED_FRAMES
from dataset import (
    collect_ballroom_pairs, collect_guitarset_pairs,
    collect_gtzan_pairs, collect_hainsworth_pairs,
    collect_smc_pairs,
    parse_beats, compute_mel,
)
import model          as _m1
import model_new_arch as _m2
from train import peak_pick

try:
    from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor
    MADMOM_AVAILABLE = True
except Exception as _madmom_err:
    MADMOM_AVAILABLE = False
    print(f"[Warning] madmom not available ({type(_madmom_err).__name__}: {_madmom_err}) — madmom scores will be skipped.")

# ── 設定 ──────────────────────────────────────────────────────────────────────
CKPT_V1    = 'checkpoints/best_model_F=0.6457.pt'
CKPT_V2    = 'checkpoints/best_model_new_arch.pt'
DATA_ROOT  = 'dataset'
CACHE_DIR  = 'dataset/mel_cache'
OUT_CSV    = 'compare_results.csv'
OUT_PNG    = 'compare_results_bargraph.png'
STRIDE     = FIXED_FRAMES // 2
TEST_RATIO = 0.1
SEED       = 42
TOLERANCE  = 0.07
TOLERANCES = [0.03, 0.05, 0.07, 0.09]   # 30 / 50 / 70 / 90 ms


# ── Test split 還原 ────────────────────────────────────────────────────────────

def get_test_pairs(data_root=DATA_ROOT):
    """以相同 seed=42 還原 BeatDataset(split='test') 的歌曲清單。"""
    ballroom   = [(p, a, 'Ballroom')   for p, a in collect_ballroom_pairs(data_root)]
    guitarset  = [(p, a, 'GuitarSet')  for p, a in collect_guitarset_pairs(data_root)]
    gtzan      = [(p, a, 'GTZAN')      for p, a in collect_gtzan_pairs(data_root)]
    hainsworth = [(p, a, 'Hainsworth') for p, a in collect_hainsworth_pairs(data_root)]
    all_pairs  = ballroom + guitarset + gtzan + hainsworth

    rng    = np.random.default_rng(SEED)
    idx    = rng.permutation(len(all_pairs))
    n_test = int(len(all_pairs) * TEST_RATIO)
    return [all_pairs[i] for i in idx[:n_test]]


# ── 推理函式 ───────────────────────────────────────────────────────────────────

def run_librosa(y):
    """Librosa beat tracker → beat_times (sec)。"""
    _, beat_frames = librosa.beat.beat_track(
        y=y, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, start_bpm=120,
    )
    return librosa.frames_to_time(beat_frames, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)


def run_beattcn(model, mel, device):
    """
    全曲滑動視窗推理（適用 v1 和 v2，beat_act 永遠是 forward 的第一個輸出）。
    重疊區域 activation 取平均後 peak-pick → beat 時間點（秒）。
    """
    T          = mel.shape[1]
    activation = np.zeros(T, dtype=np.float64)
    count      = np.zeros(T, dtype=np.float64)

    model.eval()
    with torch.no_grad():
        start = 0
        while start < T:
            end   = start + FIXED_FRAMES
            chunk = mel[:, start : min(end, T)].copy()
            if chunk.shape[1] < FIXED_FRAMES:
                chunk = np.pad(chunk, ((0, 0), (0, FIXED_FRAMES - chunk.shape[1])))
            chunk = (chunk - chunk.mean()) / (chunk.std() + 1e-6)

            x       = torch.from_numpy(chunk).unsqueeze(0).unsqueeze(0).float().to(device)
            outputs = model(x)
            act_np  = outputs[0].squeeze().cpu().numpy()   # beat_act，不管幾個輸出都取 [0]

            actual_len = min(FIXED_FRAMES, T - start)
            activation[start : start + actual_len] += act_np[:actual_len]
            count     [start : start + actual_len] += 1

            if end >= T:
                break
            start += STRIDE

    activation  = np.where(count > 0, activation / count, 0.0).astype(np.float32)
    beat_frames = peak_pick(activation)
    return beat_frames * HOP_LENGTH / SAMPLE_RATE


# ── madmom beat tracker ───────────────────────────────────────────────────────

_madmom_proc = None   # lazy-init，避免每首歌重新建立

def run_madmom(audio_path):
    """
    madmom DBNBeatTrackingProcessor → beat_times (sec)。
    使用 RNN activation + DBN decoding（Böck et al. 2014/2020 的標準做法）。
    """
    global _madmom_proc
    if _madmom_proc is None:
        _madmom_proc = DBNBeatTrackingProcessor(fps=100)
    act        = RNNBeatProcessor()(str(audio_path))
    beat_times = _madmom_proc(act)
    return np.array(beat_times, dtype=np.float32)


# ── Beat F-measure ─────────────────────────────────────────────────────────────

def fmeasure(pred_times, gt_times, tol=TOLERANCE):
    if len(pred_times) == 0 and len(gt_times) == 0:
        return 1.0
    if len(pred_times) == 0 or len(gt_times) == 0:
        return 0.0
    matched = set()
    tp = 0
    for pt in pred_times:
        dists = np.abs(gt_times - pt)
        j = int(np.argmin(dists))
        if dists[j] <= tol and j not in matched:
            tp += 1
            matched.add(j)
    precision = tp / len(pred_times)
    recall    = tp / len(gt_times)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ── Dataset distribution chart ────────────────────────────────────────────────

def plot_dataset_distribution(data_root=DATA_ROOT, out_png='dataset_distribution.png'):
    """
    左：pie chart 顯示各 dataset 佔訓練集歌曲數比例
    中：stacked bar 顯示 train/val/test 分割
    右：BPM 分布 histogram（每個 dataset 一個顏色）
    """
    collectors = {
        'Ballroom'  : collect_ballroom_pairs,
        'GuitarSet' : collect_guitarset_pairs,
        'GTZAN'     : collect_gtzan_pairs,
        'Hainsworth': collect_hainsworth_pairs,
    }

    colors = ['#5B9BD5', '#ED7D31', '#70AD47', '#FFC000']
    names  = list(collectors.keys())

    # ── 收集各 dataset 的配對 + BPM ───────────────────────────────────────────
    pairs_by_ds = {name: fn(data_root) for name, fn in collectors.items()}
    counts = {name: len(p) for name, p in pairs_by_ds.items()}
    total  = sum(counts.values())

    bpm_by_ds = {}
    print("Computing BPM distributions...")
    for name, pairs in pairs_by_ds.items():
        bpms = []
        for _, beats_path in pairs:
            ts, _ = parse_beats(beats_path)
            if len(ts) >= 2:
                bpm = float(np.clip(60.0 / np.median(np.diff(ts)), 30, 250))
                bpms.append(bpm)
        bpm_by_ds[name] = bpms

    # split 比例
    test_ratio, val_ratio = 0.1, 0.1
    sizes  = [counts[n] for n in names]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # ── 左：pie chart ─────────────────────────────────────────────────────────
    wedges, texts, autotexts = ax1.pie(
        sizes,
        labels=names,
        colors=colors,
        autopct=lambda p: f'{p:.1f}%\n({int(round(p*total/100))} songs)',
        startangle=140,
        pctdistance=0.72,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5},
    )
    for at in autotexts:
        at.set_fontsize(9)
    ax1.set_title(f'Dataset Composition\n(total {total} songs)', fontsize=12)

    # ── 中：stacked bar chart（train / val / test） ───────────────────────────
    x      = np.arange(len(names))
    n_test = [int(s * test_ratio) for s in sizes]
    n_val  = [int(s * val_ratio)  for s in sizes]
    n_tr   = [s - t - v for s, t, v in zip(sizes, n_test, n_val)]

    b1 = ax2.bar(x, n_tr,   color='#4CAF50', label='Train')
    ax2.bar(x, n_val,  bottom=n_tr, color='#FFC107', label='Val')
    ax2.bar(x, n_test, bottom=[a+b for a, b in zip(n_tr, n_val)],
            color='#F44336', label='Test')

    for bar, total_h in zip(b1, sizes):
        ax2.text(bar.get_x() + bar.get_width()/2, total_h + 3,
                 str(total_h), ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax2.set_xticks(x)
    ax2.set_xticklabels(names)
    ax2.set_ylabel('Number of Songs')
    ax2.set_title('Songs per Dataset  (train / val / test)', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax2.set_axisbelow(True)

    # ── 右：BPM histogram ────────────────────────────────────────────────────
    bins = np.arange(30, 255, 10)
    for name, color in zip(names, colors):
        ax3.hist(bpm_by_ds[name], bins=bins, alpha=0.55,
                 color=color, label=f'{name} (n={len(bpm_by_ds[name])})',
                 edgecolor='white', linewidth=0.4)

    ax3.set_xlabel('BPM')
    ax3.set_ylabel('Number of Songs')
    ax3.set_title('BPM Distribution per Dataset', fontsize=12)
    ax3.legend(fontsize=8)
    ax3.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax3.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"Dataset distribution chart saved → {Path(out_png).resolve()}")
    print(f"  Total songs : {total}")
    for name, n in counts.items():
        print(f"  {name:<12} {n:4d}  ({100*n/total:.1f}%)")


# ── Augmentation BPM distribution ─────────────────────────────────────────────

def plot_augmentation_distribution(data_root=DATA_ROOT,
                                   out_png='augmentation_distribution.png'):
    """
    上：augmentation 前的 train split BPM 分布（各 dataset 堆疊）
    下：augmentation 後（原始 + 放慢版本 ×0.75）的 BPM 分布
    """
    from dataset import SLOW_AUG_RATE, SLOW_AUG_MIN_BPM

    collectors = {
        'Ballroom'  : collect_ballroom_pairs,
        'GuitarSet' : collect_guitarset_pairs,
        'GTZAN'     : collect_gtzan_pairs,
        'Hainsworth': collect_hainsworth_pairs,
    }
    colors = ['#5B9BD5', '#ED7D31', '#70AD47', '#FFC000']
    names  = list(collectors.keys())

    # 還原 train split（同 seed=42，跳過 test+val 的前 20%）
    rng = np.random.default_rng(SEED)
    all_pairs_flat = []
    for name, fn in collectors.items():
        for p in fn(data_root):
            all_pairs_flat.append((name, p))
    idx       = rng.permutation(len(all_pairs_flat))
    n_skip    = int(len(all_pairs_flat) * (TEST_RATIO + 0.1))
    train_idx = idx[n_skip:]

    orig_bpms = {n: [] for n in names}
    aug_bpms  = {n: [] for n in names}

    for i in train_idx:
        ds_name, (_, beats_path) = all_pairs_flat[i]
        ts, _ = parse_beats(beats_path)
        if len(ts) < 2:
            continue
        bpm = float(np.clip(60.0 / np.median(np.diff(ts)), 30, 250))
        orig_bpms[ds_name].append(bpm)
        aug_bpms[ds_name].append(bpm)
        if bpm > SLOW_AUG_MIN_BPM:
            aug_bpms[ds_name].append(float(np.clip(bpm * SLOW_AUG_RATE, 30, 250)))

    bins = np.arange(30, 256, 5)

    def stacked_hist(ax, bpm_dict, title):
        bottoms = np.zeros(len(bins) - 1)
        for name, color in zip(names, colors):
            vals = np.array(bpm_dict[name])
            if len(vals) == 0:
                continue
            counts, _ = np.histogram(vals, bins=bins)
            ax.bar(bins[:-1], counts, width=5, bottom=bottoms,
                   color=color, label=name, alpha=0.85,
                   edgecolor='white', linewidth=0.3)
            bottoms += counts
        total = sum(len(v) for v in bpm_dict.values())
        ax.axvline(SLOW_AUG_MIN_BPM, color='red', linestyle='--',
                   linewidth=1.5, label=f'Aug threshold ({SLOW_AUG_MIN_BPM} BPM)')
        ax.set_title(f'{title}  (n={total})', fontsize=11)
        ax.set_ylabel('Number of Songs')
        ax.yaxis.grid(True, linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8, loc='upper right')

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    stacked_hist(ax1, orig_bpms, 'Before Augmentation  (train split, original only)')
    stacked_hist(ax2, aug_bpms,
                 f'After Augmentation  '
                 f'(+ slow-stretch ×{SLOW_AUG_RATE} for BPM > {SLOW_AUG_MIN_BPM})')

    # 箭頭：示意高 BPM 映射到低 BPM
    ymax = ax2.get_ylim()[1]
    ax2.annotate('', xy=(80, ymax * 0.55), xytext=(130, ymax * 0.55),
                 arrowprops=dict(arrowstyle='->', color='red', lw=1.8))
    ax2.text(105, ymax * 0.60, f'×{SLOW_AUG_RATE}',
             ha='center', color='red', fontsize=9, fontweight='bold')

    ax2.set_xlabel('BPM')
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"Augmentation distribution chart saved → {Path(out_png).resolve()}")


# ── Per-dataset bar chart ──────────────────────────────────────────────────────

def plot_bargraph(results, out_png=OUT_PNG):
    """4 bars per dataset group: Librosa / BeatTCN v1 / BeatTCN v2 / madmom。"""
    datasets = ['Ballroom', 'GuitarSet', 'GTZAN', 'Hainsworth']
    models   = [
        ('librosa_F', 'Librosa',              '#5B9BD5'),
        ('v1_F',      'BeatTCN v1',           '#ED7D31'),
        ('v2_F',      'BeatTCN v2 (new arch)','#70AD47'),
        ('madmom_F',  'madmom (DBN)',          '#9B59B6'),
    ]
    if not MADMOM_AVAILABLE:
        models = models[:-1]

    present = []
    data    = {key: {'means': [], 'stds': []} for key, _, _ in models}

    for ds in datasets + ['Overall']:
        rows = [r for r in results if r['dataset'] == ds] if ds != 'Overall' else results
        if not rows:
            continue
        present.append(ds)
        for key, _, _ in models:
            vals = [r[key] for r in rows]
            data[key]['means'].append(np.mean(vals))
            data[key]['stds'].append(np.std(vals))

    n_groups = len(present)
    n_models = len(models)
    width    = 0.22
    x        = np.arange(n_groups)
    offsets  = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * width

    fig, ax = plt.subplots(figsize=(11, 5))

    for (key, label, color), offset in zip(models, offsets):
        means = data[key]['means']
        stds  = data[key]['stds']
        bars  = ax.bar(x + offset, means, width, yerr=stds,
                       label=label, color=color,
                       capsize=3, error_kw={'elinewidth': 1.1})
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.013,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=7)

    # 虛線把 per-dataset 和 Overall 分開
    ax.axvline(x=n_groups - 1.5, color='gray', linestyle='--', linewidth=0.8)

    ax.set_xlabel('Dataset')
    ax.set_ylabel('Beat F-measure  (mean ± std)')
    ax.set_title('Beat Tracking Comparison  —  3 Models  (tolerance = 70 ms)')
    ax.set_xticks(x)
    ax.set_xticklabels(present)
    ax.set_ylim(0, 1.10)
    ax.legend(loc='upper left')
    ax.yaxis.grid(True, linestyle='--', alpha=0.45)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"Bar chart saved → {Path(out_png).resolve()}")


# ── Tolerance vs F-measure chart ──────────────────────────────────────────────

def plot_tolerance_fmeasure(results, out_png='tolerance_fmeasure.png',
                            datasets=None):
    """
    折線圖：X 軸為 tolerance (ms)，Y 軸為 mean Beat F-measure。
    datasets: 子圖清單（預設 4 個訓練集 + Overall）。
    """
    tol_ms   = [int(t * 1000) for t in TOLERANCES]   # [30, 50, 70, 90]
    if datasets is None:
        datasets = ['Ballroom', 'GuitarSet', 'GTZAN', 'Hainsworth', 'Overall']
    models   = [
        ('lb',     'Librosa',              '#5B9BD5', 'o'),
        ('v1',     'BeatTCN v1 (F=0.6457)','#ED7D31', 's'),
        ('v2',     'BeatTCN v2 (new arch)', '#70AD47', '^'),
        ('mm',     'madmom (DBN)',           '#9B59B6', 'D'),
    ]
    if not MADMOM_AVAILABLE:
        models = models[:-1]

    n_plots = len(datasets)
    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4.5), sharey=True)

    for ax, ds in zip(axes, datasets):
        rows = results if ds == 'Overall' else [r for r in results if r['dataset'] == ds]
        if not rows:
            ax.set_visible(False)
            continue

        for prefix, label, color, marker in models:
            means = [np.mean([r[f'{prefix}_F_{ms}'] for r in rows]) for ms in tol_ms]
            stds  = [np.std ([r[f'{prefix}_F_{ms}'] for r in rows]) for ms in tol_ms]
            ax.plot(tol_ms, means, color=color, marker=marker,
                    linewidth=2, markersize=6, label=label)
            ax.fill_between(tol_ms,
                            [m - s for m, s in zip(means, stds)],
                            [m + s for m, s in zip(means, stds)],
                            color=color, alpha=0.12)

        n = len(rows)
        ax.set_title(f'{ds}\n(n={n})', fontsize=10)
        ax.set_xlabel('Tolerance (ms)')
        ax.set_xticks(tol_ms)
        ax.set_ylim(0, 1.05)
        ax.yaxis.grid(True, linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)

    axes[0].set_ylabel('Beat F-measure (mean ± std)')
    # Overall subplot is the last one — make it slightly bolder
    axes[-1].spines['left'].set_linewidth(1.5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3,
               bbox_to_anchor=(0.5, -0.05), fontsize=9)

    fig.suptitle('Beat F-measure vs. Tolerance  —  3 Models', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Tolerance chart saved → {Path(out_png).resolve()}")


# ── SMC 專用圖表 ───────────────────────────────────────────────────────────────

def plot_smc_distribution(smc_pairs, out_png='smc_distribution.png'):
    """SMC BPM 分布 histogram（獨立一張圖）。"""
    bpms = []
    for _, beats_path in smc_pairs:
        ts, _ = parse_beats(beats_path)
        if len(ts) >= 2:
            bpm = float(np.clip(60.0 / np.median(np.diff(ts)), 30, 250))
            bpms.append(bpm)

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.arange(30, 255, 10)
    ax.hist(bpms, bins=bins, color='#9B59B6', edgecolor='white',
            linewidth=0.5, alpha=0.85)

    ax.axvline(np.mean(bpms),   color='red',    linestyle='--',
               linewidth=1.5, label=f'Mean  {np.mean(bpms):.1f} BPM')
    ax.axvline(np.median(bpms), color='orange', linestyle='-.',
               linewidth=1.5, label=f'Median {np.median(bpms):.1f} BPM')

    ax.set_xlabel('BPM')
    ax.set_ylabel('Number of Songs')
    ax.set_title(f'SMC Dataset — BPM Distribution  (n={len(bpms)})')
    ax.legend()
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"SMC distribution chart saved → {Path(out_png).resolve()}")


def plot_smc_bargraph(results, out_png='smc_bargraph.png'):
    """SMC 全集評估的獨立 bar chart（無 sub-dataset 分組）。"""
    models = [
        ('librosa_F', 'Librosa',              '#5B9BD5'),
        ('v1_F',      'BeatTCN v1 (F=0.6457)','#ED7D31'),
        ('v2_F',      'BeatTCN v2 (new arch)', '#70AD47'),
        ('madmom_F',  'madmom (DBN)',           '#9B59B6'),
    ]
    if not MADMOM_AVAILABLE:
        models = models[:-1]

    means = [np.mean([r[k] for r in results]) for k, _, _ in models]
    stds  = [np.std ([r[k] for r in results]) for k, _, _ in models]
    labels = [label for _, label, _ in models]
    colors = [color for _, _, color in models]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, means, yerr=stds, color=colors,
                  capsize=5, error_kw={'elinewidth': 1.3}, width=0.45)

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + 0.015,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Beat F-measure  (mean ± std)')
    ax.set_title(f'SMC Dataset  —  3 Models  (n={len(results)}, tolerance = 70 ms)')
    ax.set_ylim(0, 1.10)
    ax.yaxis.grid(True, linestyle='--', alpha=0.45)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"SMC bar chart saved → {Path(out_png).resolve()}")


# ── 主程式 ─────────────────────────────────────────────────────────────────────

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}\n")

    # ── Dataset distribution ───────────────────────────────────────────────────
    plot_dataset_distribution()
    plot_augmentation_distribution()

    # ── 載入兩個模型 ──────────────────────────────────────────────────────────
    model_v1 = _m1.BeatTCN().to(device)
    ckpt_v1  = torch.load(CKPT_V1, map_location=device)
    model_v1.load_state_dict(ckpt_v1['model_state'])
    print(f"[v1] {CKPT_V1}  epoch={ckpt_v1['epoch']}  val_F={ckpt_v1['val_f']:.4f}")

    model_v2 = _m2.BeatTCN().to(device)
    ckpt_v2  = torch.load(CKPT_V2, map_location=device)
    model_v2.load_state_dict(ckpt_v2['model_state'])
    print(f"[v2] {CKPT_V2}  epoch={ckpt_v2['epoch']}  val_F={ckpt_v2['val_f']:.4f}\n")

    # ── Test split ────────────────────────────────────────────────────────────
    test_pairs = get_test_pairs(DATA_ROOT)
    print(f"Test songs : {len(test_pairs)}\n")

    hdr = f"{'#':>4}  {'File':<38}  {'Dataset':<10}  {'Librosa':>7}  {'v1':>7}  {'v2':>7}  {'madmom':>7}"
    print(hdr)
    print("-" * len(hdr))

    cache_dir = Path(CACHE_DIR)
    results   = []

    for i, (wav_path, beats_path, ds_name) in enumerate(test_pairs):
        label = wav_path.name[:36]

        try:
            y, _ = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)
        except Exception as e:
            print(f"{i+1:4d}  {label:<38}  SKIP ({e})")
            continue

        gt_times, _ = parse_beats(beats_path)
        if len(gt_times) < 2:
            print(f"{i+1:4d}  {label:<38}  SKIP (too few GT beats)")
            continue

        # ── Librosa ──────────────────────────────────────────────────────────
        lb_times = run_librosa(y)
        lb_f     = fmeasure(lb_times, gt_times)

        # ── BeatTCN（mel cache 加速） ──────────────────────────────────────
        cache_file = cache_dir / (wav_path.stem + '.npy')
        mel = np.load(cache_file) if cache_file.exists() else compute_mel(wav_path)

        v1_times = run_beattcn(model_v1, mel, device)
        v1_f     = fmeasure(v1_times, gt_times)

        v2_times = run_beattcn(model_v2, mel, device)
        v2_f     = fmeasure(v2_times, gt_times)

        if MADMOM_AVAILABLE:
            mm_times = run_madmom(wav_path)
            mm_f     = fmeasure(mm_times, gt_times)
        else:
            mm_times, mm_f = np.array([]), float('nan')

        print(f"{i+1:4d}  {label:<38}  {ds_name:<10}  "
              f"{lb_f:7.4f}  {v1_f:7.4f}  {v2_f:7.4f}  {mm_f:7.4f}")

        # F-score at all tolerances
        tol_scores = {f'lb_F_{int(t*1000)}' : fmeasure(lb_times, gt_times, tol=t) for t in TOLERANCES}
        tol_scores.update({f'v1_F_{int(t*1000)}' : fmeasure(v1_times, gt_times, tol=t) for t in TOLERANCES})
        tol_scores.update({f'v2_F_{int(t*1000)}' : fmeasure(v2_times, gt_times, tol=t) for t in TOLERANCES})
        if MADMOM_AVAILABLE:
            tol_scores.update({f'mm_F_{int(t*1000)}' : fmeasure(mm_times, gt_times, tol=t) for t in TOLERANCES})

        results.append({
            'file'      : wav_path.name,
            'dataset'   : ds_name,
            'librosa_F' : round(lb_f, 4),
            'v1_F'      : round(v1_f, 4),
            'v2_F'      : round(v2_f, 4),
            'madmom_F'  : round(mm_f, 4),
            'gt_beats'  : len(gt_times),
            **{k: round(v, 4) for k, v in tol_scores.items()},
        })

    if not results:
        print("No results.")
        return

    # ── 統計摘要 ───────────────────────────────────────────────────────────────
    lb  = np.array([r['librosa_F'] for r in results])
    v1  = np.array([r['v1_F']      for r in results])
    v2  = np.array([r['v2_F']      for r in results])
    mm  = np.array([r['madmom_F']  for r in results])
    n   = len(results)

    print(f"\n{'='*68}")
    print(f"{'Model':<28} {'Mean F':>8} {'Median':>8} {'Std':>7} {'Songs':>6}")
    print(f"{'-'*68}")
    print(f"{'Librosa baseline':<28} {lb.mean():8.4f} {np.median(lb):8.4f} {lb.std():7.4f} {n:6d}")
    print(f"{'BeatTCN v1 (F=0.6457)':<28} {v1.mean():8.4f} {np.median(v1):8.4f} {v1.std():7.4f} {n:6d}")
    print(f"{'BeatTCN v2 (new arch)':<28} {v2.mean():8.4f} {np.median(v2):8.4f} {v2.std():7.4f} {n:6d}")
    if MADMOM_AVAILABLE:
        valid = mm[~np.isnan(mm)]
        print(f"{'madmom (DBN)':<28} {np.nanmean(mm):8.4f} {np.nanmedian(mm):8.4f} {np.nanstd(mm):7.4f} {len(valid):6d}")
    print(f"{'='*68}")

    # ── 存 CSV ────────────────────────────────────────────────────────────────
    out = Path(OUT_CSV)
    with open(out, 'w', newline='', encoding='utf-8') as f:
        prefixes   = ['lb', 'v1', 'v2'] + (['mm'] if MADMOM_AVAILABLE else [])
        tol_fields = [f'{p}_F_{int(t*1000)}' for t in TOLERANCES for p in prefixes]
        writer = csv.DictWriter(f, fieldnames=[
            'file', 'dataset', 'librosa_F', 'v1_F', 'v2_F', 'madmom_F', 'gt_beats', *tol_fields,
        ])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nCSV saved → {out.resolve()}")

    # ── Bar chart ─────────────────────────────────────────────────────────────
    plot_bargraph(results)
    plot_tolerance_fmeasure(results)

    # ── SMC 評估（完全 held-out，不在訓練集內） ────────────────────────────────
    smc_pairs = collect_smc_pairs(DATA_ROOT)
    if not smc_pairs:
        print("\n[SMC] 找不到 SMC 音訊，跳過。")
        return

    print(f"\n{'='*62}")
    print(f"SMC Evaluation  ({len(smc_pairs)} songs, fully held-out)")
    print(f"{'='*62}")
    hdr2 = f"{'#':>4}  {'File':<20}  {'Librosa':>7}  {'v1':>7}  {'v2':>7}"
    print(hdr2)
    print("-" * len(hdr2))

    smc_results = []
    for i, (wav_path, beats_path) in enumerate(smc_pairs):
        try:
            y, _ = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)
        except Exception as e:
            print(f"{i+1:4d}  {wav_path.name:<20}  SKIP ({e})")
            continue

        gt_times, _ = parse_beats(beats_path)
        if len(gt_times) < 2:
            continue

        lb_times = run_librosa(y)
        lb_f     = fmeasure(lb_times, gt_times)

        cache_file = cache_dir / (wav_path.stem + '.npy')
        mel = np.load(cache_file) if cache_file.exists() else compute_mel(wav_path)

        v1_times = run_beattcn(model_v1, mel, device)
        v1_f     = fmeasure(v1_times, gt_times)

        v2_times = run_beattcn(model_v2, mel, device)
        v2_f     = fmeasure(v2_times, gt_times)

        if MADMOM_AVAILABLE:
            mm_times = run_madmom(wav_path)
            mm_f     = fmeasure(mm_times, gt_times)
        else:
            mm_times, mm_f = np.array([]), float('nan')

        tol_scores = {f'lb_F_{int(t*1000)}': fmeasure(lb_times, gt_times, tol=t) for t in TOLERANCES}
        tol_scores.update({f'v1_F_{int(t*1000)}': fmeasure(v1_times, gt_times, tol=t) for t in TOLERANCES})
        tol_scores.update({f'v2_F_{int(t*1000)}': fmeasure(v2_times, gt_times, tol=t) for t in TOLERANCES})
        if MADMOM_AVAILABLE:
            tol_scores.update({f'mm_F_{int(t*1000)}': fmeasure(mm_times, gt_times, tol=t) for t in TOLERANCES})

        print(f"{i+1:4d}  {wav_path.name:<20}  {lb_f:7.4f}  {v1_f:7.4f}  {v2_f:7.4f}  {mm_f:7.4f}")
        smc_results.append({
            'file'      : wav_path.name,
            'dataset'   : 'SMC',
            'librosa_F' : round(lb_f, 4),
            'v1_F'      : round(v1_f, 4),
            'v2_F'      : round(v2_f, 4),
            'madmom_F'  : round(mm_f, 4),
            'gt_beats'  : len(gt_times),
            **{k: round(v, 4) for k, v in tol_scores.items()},
        })

    if smc_results:
        lb  = np.array([r['librosa_F'] for r in smc_results])
        v1  = np.array([r['v1_F']      for r in smc_results])
        v2  = np.array([r['v2_F']      for r in smc_results])
        ns  = len(smc_results)
        print(f"\n{'='*62}")
        print(f"{'Model':<24} {'Mean F':>8} {'Median':>8} {'Std':>7} {'Songs':>6}")
        print(f"{'-'*62}")
        print(f"{'Librosa baseline':<24} {lb.mean():8.4f} {np.median(lb):8.4f} {lb.std():7.4f} {ns:6d}")
        print(f"{'BeatTCN v1 (F=0.6457)':<24} {v1.mean():8.4f} {np.median(v1):8.4f} {v1.std():7.4f} {ns:6d}")
        print(f"{'BeatTCN v2 (new arch)':<24} {v2.mean():8.4f} {np.median(v2):8.4f} {v2.std():7.4f} {ns:6d}")
        print(f"{'='*62}")

        smc_csv = Path('smc_results.csv')
        with open(smc_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'file', 'dataset', 'librosa_F', 'v1_F', 'v2_F', 'gt_beats',
            ])
            writer.writeheader()
            writer.writerows(smc_results)
        print(f"SMC CSV saved → {smc_csv.resolve()}")

        plot_smc_bargraph(smc_results)
        plot_smc_distribution(smc_pairs)
        plot_tolerance_fmeasure(smc_results,
                                out_png='smc_tolerance_fmeasure.png',
                                datasets=['SMC', 'Overall'])


if __name__ == '__main__':
    main()
