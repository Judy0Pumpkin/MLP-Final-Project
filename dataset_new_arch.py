import os
os.environ['NUMBA_DISABLE_SVML'] = '1'   # prevent LLVM crash on Windows

import numpy as np
import librosa
import torch
from torch.utils.data import Dataset
from pathlib import Path
from scipy.ndimage import gaussian_filter1d

from constants import (
    SAMPLE_RATE, N_MELS, HOP_LENGTH, N_FFT,
    F_MIN, F_MAX, FIXED_FRAMES, TEMPO_MIN, TEMPO_MAX,
)

BEAT_SIGMA   = 1.5   # Gaussian smoothing for beat labels (frames)
N_TEMPO_BINS = 300   # log-spaced BPM classification bins

# 多目標 BPM augmentation：對每首歌生成能拉到各目標 BPM 的 stretch 版本
# 使訓練集的 BPM 分布接近 uniform distribution（70-160 BPM）
AUG_TARGET_BPMS = list(range(60, 170, 5))    # [60, 65, 70, ..., 165] 每 5 BPM 一個目標
AUG_RATE_MIN    = 0.5    # stretch rate 下限
AUG_RATE_MAX    = 2.0    # stretch rate 上限
AUG_RATE_SKIP   = 0.1    # |rate-1.0| < 此值時跳過


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_beats(beats_path):
    """
    .beats 檔案格式: <timestamp_sec>  <beat_position(1-4)>，空白或 tab 分隔
        0.1944 4
        0.8499 1
    回傳 (timestamps, positions) numpy array
    """
    timestamps, positions = [], []
    with open(beats_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    timestamps.append(float(parts[0]))
                    positions.append(int(float(parts[1])))
                except ValueError:
                    continue
    return np.array(timestamps, dtype=np.float32), np.array(positions, dtype=np.int32)


def make_beat_activation(timestamps, n_frames, sigma=BEAT_SIGMA):
    """把離散的拍點轉換成連續的 activation，並做 Gaussian smoothing。
    __|___|___|___|___|___(拍點)
    __/^\____/^\____/^\__(可能是拍點位置的機率分布)
    """
    frame_idx = librosa.time_to_frames(
        timestamps, sr=SAMPLE_RATE, hop_length=HOP_LENGTH
    )
    frame_idx = frame_idx[(frame_idx >= 0) & (frame_idx < n_frames)]

    activation = np.zeros(n_frames, dtype=np.float32)
    activation[frame_idx] = 1.0
    activation = gaussian_filter1d(activation, sigma=sigma)
    if activation.max() > 0:
        activation /= activation.max()
    return activation


def bpm_to_bin(bpm):
    """BPM → log-spaced bin index (0 … N_TEMPO_BINS-1)
    用 log 空間是因為人類對節奏的感知接近對數尺度
    （60→120 BPM 感覺差很多，120→180 差異感覺較小）
    """
    bpm = np.clip(float(bpm), TEMPO_MIN, TEMPO_MAX)
    log_ratio = np.log(bpm / TEMPO_MIN) / np.log(TEMPO_MAX / TEMPO_MIN)
    return int(np.clip(log_ratio * (N_TEMPO_BINS - 1), 0, N_TEMPO_BINS - 1))
##Böck 2020用的是雙峰目標：主要 BPM 給 1.0，倍速/半速給 0.5）
def bpm_to_tempo_target(bpm):
    """回傳 soft tempo target，主BPM=1.0，倍速/半速=0.5"""
    target = np.zeros(N_TEMPO_BINS, dtype=np.float32)
    for weight, b in [(1.0, bpm), (0.5, bpm*2), (0.5, bpm/2)]:
        b = np.clip(b, TEMPO_MIN, TEMPO_MAX)
        idx = bpm_to_bin(b)
        target[idx] = max(target[idx], weight)
    return target

def compute_mel(audio_path, stretch_rate=1.0):
    """Load WAV 或 MP3，可選做 time stretch，轉換成 power-dB mel spectrogram。
    回傳 shape (N_MELS, T) float32。

    mel 的結構類似於一個 2D 表格：
        行 = N_MELS 個 mel 頻率 bin（從低頻到高頻）
        欄 = T 個時間 frame，每 frame ≈ hop_length/sr ≈ 23ms
        數值 = 該頻率在該時間的能量 (dB)

    Args:
        stretch_rate: 1.0 = 原速；0.75 = 放慢到 75%（beat timestamps 需除以 0.75）
                      需要 NUMBA_DISABLE_SVML=1 才能在 Windows 上使用 stretch
    """
    y, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    if stretch_rate != 1.0:
        y = librosa.effects.time_stretch(y, rate=stretch_rate)
    mel = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=F_MIN, fmax=F_MAX,
    )
    mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    return mel


# ── Dataset pair collectors ───────────────────────────────────────────────────
# 每個函式回傳 list of (audio_path, beats_path) 配對

def collect_ballroom_pairs(data_root):
    """Ballroom: BallroomData/**/*.wav ↔ beat_this_annotations/ballroom/.../ballroom_STEM.beats"""
    data_root = Path(data_root)
    beats_dir = data_root / 'beat_this_annotations' / 'ballroom' / 'annotations' / 'beats'
    pairs = []
    for wav in sorted((data_root / 'BallroomData').rglob('*.wav')):
        ann = beats_dir / ('ballroom_' + wav.stem + '.beats')
        if ann.exists():
            pairs.append((wav, ann))
    return pairs


def collect_guitarset_pairs(data_root):
    """GuitarSet: GuitarSet/**/*_comp_mix.wav ↔ beat_this_annotations/guitarset/.../STEM.beats
    solo_mix 檔案沒有對應 annotation，只用 comp_mix。
    """
    data_root = Path(data_root)
    beats_dir = data_root / 'beat_this_annotations' / 'guitarset' / 'annotations' / 'beats'
    if not beats_dir.exists():
        return []
    pairs = []
    for wav in sorted((data_root / 'GuitarSet').rglob('*_comp_mix.wav')):
        ann = beats_dir / (wav.stem + '.beats')
        if ann.exists():
            pairs.append((wav, ann))
    return pairs


def collect_hainsworth_pairs(data_root):
    """Hainsworth: hainsworth/NNN.wav ↔ beat_this_annotations/hainsworth/.../hainsworth_NNN.beats"""
    data_root = Path(data_root)
    beats_dir = data_root / 'beat_this_annotations' / 'hainsworth' / 'annotations' / 'beats'
    if not beats_dir.exists():
        return []
    pairs = []
    for wav in sorted((data_root / 'hainsworth').rglob('*.wav')):
        ann = beats_dir / ('hainsworth_' + wav.stem + '.beats')
        if ann.exists():
            pairs.append((wav, ann))
    return pairs


def collect_gtzan_pairs(data_root):
    """GTZAN: GTZAN_Dataset/**/*.wav ↔ beat_this_annotations/gtzan/.../gtzan_STEM.beats
    GTZAN 的 wav 檔名用 '.'（如 blues.00000），annotation 用 '_'（如 gtzan_blues_00000）。
    """
    data_root = Path(data_root)
    beats_dir = data_root / 'beat_this_annotations' / 'gtzan' / 'annotations' / 'beats'
    if not beats_dir.exists():
        return []
    pairs = []
    for wav in sorted((data_root / 'GTZAN_Dataset').rglob('*.wav')):
        ann_stem = 'gtzan_' + wav.stem.replace('.', '_')
        ann = beats_dir / (ann_stem + '.beats')
        if ann.exists():
            pairs.append((wav, ann))
    return pairs


# ── Dataset ───────────────────────────────────────────────────────────────────

class BeatDataset(Dataset):
    """
    多資料集合併的 beat tracking Dataset。
    自動載入 Ballroom、GuitarSet、GTZAN、Hainsworth 四個資料集。

    每個 sample 是固定長度的 sliding window：
        mel   : (1, N_MELS, context_frames)  float32
        beat  : (context_frames,)             float32   0~1 activation
        tempo : ()                            int64     BPM bin index

    Args:
        data_root      : dataset/ 資料夾路徑
        split          : 'train' | 'val' | 'test'
        context_frames : 每個 sample 的 mel frame 數（預設 FIXED_FRAMES）
        stride         : sliding window 步長（預設 context_frames // 2）
        val_ratio      : val 佔總資料比例
        test_ratio     : test 佔總資料比例
        seed           : 隨機種子，確保 split 可重現
        cache_dir      : mel spectrogram 的 .npy 暫存路徑
    """

    def __init__(
        self,
        data_root,
        split          = 'train',
        context_frames = FIXED_FRAMES,
        stride         = None,
        val_ratio      = 0.1,
        test_ratio     = 0.1,
        seed           = 42,
        cache_dir      = None,
    ):
        self.split          = split
        self.context_frames = context_frames
        self.stride         = stride if stride is not None else context_frames // 2
        self.cache_dir      = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # ── 收集所有資料集的配對 ──────────────────────────────────────────────
        data_root = Path(data_root)
        all_pairs = (
            collect_ballroom_pairs(data_root) +
            collect_guitarset_pairs(data_root) +
            collect_gtzan_pairs(data_root) +
            collect_hainsworth_pairs(data_root)
        )

        # ── 依檔案做 train / val / test split ────────────────────────────────
        rng    = np.random.default_rng(seed)
        idx    = rng.permutation(len(all_pairs))
        n_test = int(len(all_pairs) * test_ratio)
        n_val  = int(len(all_pairs) * val_ratio)

        if split == 'test':
            chosen = idx[:n_test]
        elif split == 'val':
            chosen = idx[n_test : n_test + n_val]
        else:
            chosen = idx[n_test + n_val :]

        self.pairs = [all_pairs[i] for i in chosen]

        # ── 載入所有音訊 + 製作 labels ────────────────────────────────────────
        print(f"[BeatDataset] loading {split} split ({len(self.pairs)} songs) …")
        self._bpms       = []
        self._mels       = []
        self._beat_acts  = []
        self._tempo_bins = []
        self._downbeat_acts = []
    
        n_augmented = 0
        for wav_path, beats_path in self.pairs:
            try:
                mel = self._get_mel(wav_path)
            except Exception as e:
                print(f"  [跳過] {wav_path.name} — {e}")
                continue
            n_frames = mel.shape[1]

            timestamps, positions = parse_beats(beats_path)
            beat_act = make_beat_activation(timestamps, n_frames)

            if len(timestamps) >= 2:
                bpm = float(np.clip(
                    60.0 / np.median(np.diff(timestamps)), TEMPO_MIN, TEMPO_MAX
                ))
            else:
                bpm = 120.0

            self._mels.append(mel)
            self._beat_acts.append(beat_act)
            self._tempo_bins.append(bpm_to_bin(bpm))
            self._bpms.append(bpm)
            # downbeat = position == 1 的拍點
            downbeat_times = timestamps[positions == 1]
            downbeat_act = make_beat_activation(downbeat_times, n_frames, sigma=BEAT_SIGMA)
            self._downbeat_acts.append(downbeat_act)
            # ── 多目標 BPM augmentation（僅 train split） ────────────────────────
            if split == 'train':
                for target in AUG_TARGET_BPMS:
                    rate = target / bpm
                    if not (AUG_RATE_MIN <= rate <= AUG_RATE_MAX):
                        continue
                    if abs(rate - 1.0) < AUG_RATE_SKIP:
                        continue
                    try:
                        mel_aug  = self._get_mel(wav_path, rate)
                        ts_aug   = timestamps / rate
                        act_aug  = make_beat_activation(ts_aug, mel_aug.shape[1])
                        bpm_aug  = float(np.clip(bpm * rate, TEMPO_MIN, TEMPO_MAX))
                        self._mels.append(mel_aug)
                        self._beat_acts.append(act_aug)
                        self._tempo_bins.append(bpm_to_bin(bpm_aug))
                        self._bpms.append(bpm_aug)
                        ts_db_aug = timestamps[positions == 1] / rate
                        self._downbeat_acts.append(
                            make_beat_activation(ts_db_aug, mel_aug.shape[1]))
                        n_augmented += 1
                    except Exception as e:
                        print(f"  [aug 跳過] {wav_path.name} rate={rate:.2f} — {e}")

        if n_augmented:
            print(f"  slow-stretch augmented: {n_augmented} songs added")

        # ── 製作 Sliding Window Index ─────────────────────────────────────────
        # 每首歌切出多個 context_frames 長的 window，window 之間間隔 stride
        self.windows = []
        for file_idx, mel in enumerate(self._mels):
            start = 0
            while start + self.context_frames <= mel.shape[1]:
                self.windows.append((file_idx, start))
                start += self.stride

        print(f"[BeatDataset] {split}: {len(self.windows)} windows "
              f"(context={self.context_frames} frames, stride={self.stride})")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _get_mel(self, audio_path, stretch_rate=1.0):
        """Cache-aware mel loader：有 .npy 快取就直接讀，否則算完再存。
        rate=1.0 → stem.npy；rate=0.75 → stem_s075.npy
        """
        if self.cache_dir is not None:
            if stretch_rate == 1.0:
                cache_file = self.cache_dir / (Path(audio_path).stem + '.npy')
            else:
                tag        = f"s{int(stretch_rate * 100):03d}"
                cache_file = self.cache_dir / f"{Path(audio_path).stem}_{tag}.npy"

            if cache_file.exists():
                return np.load(cache_file)
            mel = compute_mel(audio_path, stretch_rate)
            np.save(cache_file, mel)
            return mel
        return compute_mel(audio_path, stretch_rate)

    # ── PyTorch API ───────────────────────────────────────────────────────────

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        file_idx, start = self.windows[idx]
        end = start + self.context_frames

        mel_window  = self._mels[file_idx][:, start:end].copy()
        beat_window = self._beat_acts[file_idx][start:end].copy()
        tempo_bin   = self._tempo_bins[file_idx]
        bpm         = self._bpms[file_idx]
        # ── Pitch shift augmentation（僅 train，對 mel 做頻率軸平移）──
        # pitch shift ±2 個 mel bin，beat timestamps 不變，tempo bin 不變
        if self.split == 'train' and np.random.random() < 0.5:
            shift = np.random.randint(-2, 3)  # -2, -1, 0, 1, 2
            if shift > 0:
                mel_window = np.concatenate([
                    mel_window[shift:, :],
                    np.full((shift, mel_window.shape[1]), mel_window.min())
                ], axis=0)
            elif shift < 0:
                mel_window = np.concatenate([
                    np.full((-shift, mel_window.shape[1]), mel_window.min()),
                    mel_window[:shift, :]
                ], axis=0)
        # 最後一個 window 可能不足 context_frames，補 0
        if mel_window.shape[1] < self.context_frames:
            pad = self.context_frames - mel_window.shape[1]
            mel_window  = np.pad(mel_window,  ((0, 0), (0, pad)))
            beat_window = np.pad(beat_window, (0, pad))

        # Per-window z-score 正規化（與 baseline.py 的 audio_to_mel 一致）
        mean, std  = mel_window.mean(), mel_window.std() + 1e-6
        mel_window = (mel_window - mean) / std

        downbeat_window = self._downbeat_acts[file_idx][start:end].copy()
        if downbeat_window.shape[0] < self.context_frames:
            downbeat_window = np.pad(downbeat_window, (0, self.context_frames - downbeat_window.shape[0]))

        return (
            torch.from_numpy(mel_window).unsqueeze(0).float(),
            torch.from_numpy(beat_window).float(),
            torch.from_numpy(downbeat_window).float(),
            torch.from_numpy(bpm_to_tempo_target(bpm)).float(),
        )


# ── 向後相容：保留舊名稱 ──────────────────────────────────────────────────────
BallroomDataset = BeatDataset


# ── Smoke-test ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    data_root = sys.argv[1] if len(sys.argv) > 1 else 'dataset'
    cache_dir = 'dataset/mel_cache'

    for split in ('train', 'val', 'test'):
        ds = BeatDataset(data_root, split=split, cache_dir=cache_dir)
        mel, beat, tempo = ds[0]
        print(f"  {split:5s}  mel={tuple(mel.shape)}  beat={tuple(beat.shape)}  "
              f"tempo_bin={tempo.argmax().item()}  windows={len(ds)}")