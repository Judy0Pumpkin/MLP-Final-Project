"""
分析各資料集的 BPM 分布（原始 + augmentation 後）
"""
import numpy as np
from pathlib import Path
from dataset import (
    collect_ballroom_pairs,
    collect_guitarset_pairs,
    collect_gtzan_pairs,
    collect_hainsworth_pairs,
    parse_beats,
    AUG_TARGET_BPMS,
    AUG_RATE_MIN,
    AUG_RATE_MAX,
    AUG_RATE_SKIP,
)
from constants import TEMPO_MIN, TEMPO_MAX

DATA_ROOT = Path('dataset')


def get_bpm(beats_path):
    """從 .beats 檔案計算 BPM（用 beat 間距中位數）。"""
    timestamps, _ = parse_beats(beats_path)
    if len(timestamps) < 2:
        return None
    median_ibi = float(np.median(np.diff(timestamps)))
    if median_ibi <= 0:
        return None
    return 60.0 / median_ibi


HIST_BINS = [30, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 180, 200, 250]
BAR_WIDTH  = 40


def print_hist(bpms, bar_width=BAR_WIDTH):
    counts, edges = np.histogram(bpms, bins=HIST_BINS)
    max_c = max(counts) if counts.max() > 0 else 1
    for i, c in enumerate(counts):
        bar = '█' * (c * bar_width // max_c)
        print(f"    {edges[i]:3.0f}-{edges[i+1]:3.0f} BPM : {bar} {c}")


def simulate_augmented(bpms_orig):
    """對原始 BPM 陣列，模擬套用 AUG_TARGET_BPMS 後會新增哪些 BPM。
    回傳 (原始 BPM list, 所有 augmented BPM list)。
    """
    aug_bpms = []
    for bpm in bpms_orig:
        for target in AUG_TARGET_BPMS:
            rate = target / bpm
            if not (AUG_RATE_MIN <= rate <= AUG_RATE_MAX):
                continue
            if abs(rate - 1.0) < AUG_RATE_SKIP:
                continue
            aug_bpms.append(float(np.clip(bpm * rate, TEMPO_MIN, TEMPO_MAX)))
    return aug_bpms


def analyze(name, pairs):
    bpms = [get_bpm(b) for _, b in pairs]
    bpms = [b for b in bpms if b is not None]
    bpms = np.array(bpms)

    print(f"\n── {name} ({len(bpms)} songs) ──────────────────────")
    print(f"  BPM range  : {bpms.min():.1f} ~ {bpms.max():.1f}")
    print(f"  mean / std : {bpms.mean():.1f} ± {bpms.std():.1f}")
    print(f"  median     : {np.median(bpms):.1f}")
    print("  原始分布：")
    print_hist(bpms)

    return bpms


# ── 各資料集分析 ───────────────────────────────────────────────
all_bpms = []

for name, fn in [
    ("Ballroom",   collect_ballroom_pairs),
    ("GuitarSet",  collect_guitarset_pairs),
    ("GTZAN",      collect_gtzan_pairs),
    ("Hainsworth", collect_hainsworth_pairs),
]:
    pairs = fn(DATA_ROOT)
    bpms  = analyze(name, pairs)
    all_bpms.append((name, bpms))

# ── 合計（原始） ───────────────────────────────────────────────
combined = np.concatenate([b for _, b in all_bpms])
print(f"\n══ 全部原始 ({len(combined)} songs) ════════════════════")
print(f"  BPM range  : {combined.min():.1f} ~ {combined.max():.1f}")
print(f"  mean / std : {combined.mean():.1f} ± {combined.std():.1f}")
print(f"  median     : {np.median(combined):.1f}")
print("  分布：")
print_hist(combined)

# ── Augmentation 後模擬分布 ────────────────────────────────────
aug_bpms  = simulate_augmented(combined)
all_after = np.concatenate([combined, aug_bpms])
print(f"\n══ Augmentation 後 ({len(all_after)} samples，"
      f"原始 {len(combined)} + aug {len(aug_bpms)}) ════════════")
print(f"  BPM range  : {all_after.min():.1f} ~ {all_after.max():.1f}")
print(f"  mean / std : {all_after.mean():.1f} ± {all_after.std():.1f}")
print(f"  median     : {np.median(all_after):.1f}")
print(f"  目標 BPMs  : {AUG_TARGET_BPMS}")
print("  分布：")
print_hist(all_after)

# ── 只看 aug 新增的部分 ────────────────────────────────────────
print(f"\n══ 僅 Aug 新增 ({len(aug_bpms)} samples) ════════════════")
print("  分布：")
print_hist(np.array(aug_bpms))

# ── 存成圖片 ──────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plot_bins = np.arange(30, 260, 10)   # 10 BPM 每格，更細緻

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
fig.suptitle('BPM Distribution: Original vs Augmented', fontsize=14, fontweight='bold')

axes[0].hist(combined,          bins=plot_bins, color='steelblue',  edgecolor='white', linewidth=0.4)
axes[0].set_title(f'Original ({len(combined)} songs)')
axes[0].set_ylabel('Count')

axes[1].hist(np.array(aug_bpms), bins=plot_bins, color='darkorange', edgecolor='white', linewidth=0.4)
axes[1].set_title(f'Augmented only ({len(aug_bpms)} samples)  —  targets: {AUG_TARGET_BPMS}')
axes[1].set_ylabel('Count')

axes[2].hist(all_after,          bins=plot_bins, color='seagreen',   edgecolor='white', linewidth=0.4)
axes[2].set_title(f'Original + Augmented ({len(all_after)} samples)')
axes[2].set_ylabel('Count')
axes[2].set_xlabel('BPM')

for ax in axes:
    ax.set_xlim(30, 250)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
out = Path('augmentation_distribution.png')
plt.savefig(out, dpi=150)
print(f"\n[saved] {out.resolve()}")
