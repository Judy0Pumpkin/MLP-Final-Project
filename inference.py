"""
使用訓練好的 BeatTCN 模型進行即時拍點偵測。
流程與 baseline.py 相同，只把第三步換成 BeatTCN。

    SoundAPI → audio_to_mel → BeatTCN → peak picking → 印出 beat 時間點 + BPM
"""

import numpy as np
import librosa
import torch
import sounddevice as sd

from constants import (
    SAMPLE_RATE, N_MELS, HOP_LENGTH, N_FFT,
    F_MIN, F_MAX, FIXED_FRAMES, WINDOW_SECONDS,
)
from model import BeatTCN
from train import peak_pick   # 複用 train.py 裡的 peak_pick（threshold=0.4, min_dist=5）

CHECKPOINT = 'checkpoints/best_model_new_arch.pt'


# ────────────────────────────────────────────────────────────
# 1. 音訊輸入（與 baseline.py 相同）
# ────────────────────────────────────────────────────────────

class SoundAPI:
    def __init__(self, sr=SAMPLE_RATE, duration=WINDOW_SECONDS):
        self.sr       = sr
        self.duration = duration

    def get_audio(self):
        """錄 WINDOW_SECONDS 秒的音訊，回傳 1D numpy array。"""
        audio = sd.rec(int(self.sr * self.duration),
                       samplerate=self.sr, channels=1)
        sd.wait()
        return audio.flatten()


# ────────────────────────────────────────────────────────────
# 2. Mel Spectrogram（與訓練時完全一致）
# ────────────────────────────────────────────────────────────

def audio_to_mel(y):
    """
    raw waveform → mel spectrogram tensor，preprocessing 與 dataset.py 的 compute_mel 一致。
    回傳 shape: (1, 1, N_MELS, FIXED_FRAMES)，可直接送進模型。
    """
    mel = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=F_MIN, fmax=F_MAX,
    )
    mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)

    # 長度對齊 FIXED_FRAMES
    if mel.shape[1] < FIXED_FRAMES:
        mel = np.pad(mel, ((0, 0), (0, FIXED_FRAMES - mel.shape[1])))
    else:
        mel = mel[:, :FIXED_FRAMES]

    # per-window z-score（與 dataset.py __getitem__ 一致）
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)

    return torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).float()
    # shape: (1, 1, N_MELS, FIXED_FRAMES)


# ────────────────────────────────────────────────────────────
# 3. BeatTCN 推理
# ────────────────────────────────────────────────────────────

def estimate_bpm_autocorr(beat_act, min_bpm=40, max_bpm=250):
    """
    自相關法估算 BPM：不依賴 peak picking，直接從 activation 曲線找主週期。

    原理：
        beat activation 是一個週期性訊號（每隔固定 frame 出現一個峰值）。
        把訊號和自己做相關計算，相關值在「週期整數倍」的 lag 上會出現峰值。
        找到第一個（最小 lag）的峰值，就是 beat 週期，換算成 BPM。

        lag=0    lag=1    ...  lag=T    lag=2T   ...
        corr=max         ...  corr=高  corr=高  ...
                                ↑ 這個 T 就是 beat 週期（frames）
    """
    fps     = SAMPLE_RATE / HOP_LENGTH          # frames per second ≈ 43
    min_lag = max(1, int(60.0 / max_bpm * fps)) # 最快 BPM 對應的最小 lag
    max_lag = int(60.0 / min_bpm * fps)         # 最慢 BPM 對應的最大 lag
    max_lag = min(max_lag, len(beat_act) - 1)

    if min_lag >= max_lag:
        return 0.0

    # 自相關（只取正 lag 部分）
    n    = len(beat_act)
    corr = np.correlate(beat_act, beat_act, mode='full')[n - 1:]
    # 對倍速/半速區域的 corr 做降權，避免 octave error
    for lag in range(min_lag, max_lag + 1):
        for other_lag in range(min_lag, lag):
            if lag % other_lag == 0:
                corr[lag] *= 0.5
                break
    
    # 在有效 BPM 範圍內找最大相關 lag
    best_lag = int(np.argmax(corr[min_lag:max_lag + 1])) + min_lag
    bpm      = 60.0 / (best_lag / fps)
    return float(bpm)


def run_beat_tcn(model, mel_tensor, device):
    """
    模型推理：mel tensor → beat 時間點 + 估計 BPM。


    回傳:
        beat_times : np.array，beat 出現的時間（秒）
        bpm        : float，自相關法估算的 BPM
        beat_act   : np.array (FIXED_FRAMES,)，原始 activation 曲線（供視覺化）
    """
    model.eval()
    with torch.no_grad():
        beat_act_tensor, _ = model(mel_tensor.to(device))

    beat_act = beat_act_tensor.squeeze().cpu().numpy()  # (FIXED_FRAMES,)

    # Peak picking → beat 時間點
    beat_frames = peak_pick(beat_act)
    beat_times  = beat_frames * HOP_LENGTH / SAMPLE_RATE

    # 自相關法估算 BPM（不依賴 peak picking）
    bpm = estimate_bpm_autocorr(beat_act)

    return beat_times, bpm, beat_act


# ────────────────────────────────────────────────────────────
# 4. 主程式
# ────────────────────────────────────────────────────────────

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 載入模型
    model = BeatTCN().to(device)
    ckpt  = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    print(f"=== BeatTCN Inference ===")
    print(f"Checkpoint : {CHECKPOINT}  (trained epoch {ckpt['epoch']}, val_F={ckpt['val_f']:.4f})")
    print(f"SR={SAMPLE_RATE}, HOP_LENGTH={HOP_LENGTH}, N_MELS={N_MELS}, FIXED_FRAMES={FIXED_FRAMES}")
    print(f"Device     : {device}\n")

    api  = SoundAPI()
    loop = 0

    while True:
        loop += 1
        print(f"── Round {loop} ──────────────────────")

        # Step 1: 錄音
        y = api.get_audio()
        print(f"  音訊 shape : {y.shape}")
        print(f"  音訊振幅   : max={y.max():.4f}, mean={np.abs(y).mean():.5f}")

        if np.abs(y).mean() < 1e-4:
            print("  [靜音，跳過]\n")
            continue

        # Step 2: Mel Spectrogram
        mel_tensor = audio_to_mel(y)
        print(f"  Mel shape  : {mel_tensor.shape}")

        # Step 3: BeatTCN
        beat_times, bpm, beat_act = run_beat_tcn(model, mel_tensor, device)
        print(f"  BPM        : {bpm:.1f}")
        print(f"  Beat 數量  : {len(beat_times)}")
        print(f"  Beat 時間點: {np.round(beat_times, 2)}")
        print(f"  Activation : min={beat_act.min():.3f}  max={beat_act.max():.3f}  "
              f"mean={beat_act.mean():.3f}")
        print()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n停止。")
