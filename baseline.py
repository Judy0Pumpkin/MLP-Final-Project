"""

用 librosa 的 beat_track 把整個 pipeline 跑通：
    SoundAPI → audio_to_mel → librosa baseline → 印出 BPM + beat 位置

這是確認流程正確的暫時版本，之後會換成 TCN 模型。

"""

import numpy as np
import librosa
import torch

from constants import (
    SAMPLE_RATE, N_MELS, HOP_LENGTH, N_FFT,
    F_MIN, F_MAX, FIXED_FRAMES,
    WINDOW_SECONDS, TEMPO_MIN, TEMPO_MAX,
)


# ────────────────────────────────────────────────────────────
# 1. 音訊輸入（copied from sounddeviceAPI.py）
# ────────────────────────────────────────────────────────────

import sounddevice as sd
from constants import SAMPLE_RATE, WINDOW_SECONDS

class SoundAPI:
    def __init__(self, sr=SAMPLE_RATE, duration=WINDOW_SECONDS):
        self.sr = sr
        self.duration = duration

    def get_audio(self):
        audio = sd.rec(int(self.sr * self.duration),
                       samplerate=self.sr,
                       channels=1)

        sd.wait()

        return audio.flatten()


# ────────────────────────────────────────────────────────────
# 2. Mel Spectrogram 轉換（copied from Wav_to_Mel_soundAPI.py）
# ────────────────────────────────────────────────────────────
def audio_to_mel(y, sr=SAMPLE_RATE, n_mels=N_MELS, fixed_frames=FIXED_FRAMES):

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels) #把原始音訊轉成Mel spectrogram
    mel = librosa.power_to_db(mel, ref=np.max) #分貝db
    mel = (mel - mel.mean()) / (mel.std() + 1e-6) #標準化

    if mel.shape[1] < fixed_frames: #如果資料時間長度太短需補0(padding)
        mel = np.pad(mel, ((0,0),(0,fixed_frames - mel.shape[1])), mode="constant")
    else:
        mel = mel[:, :fixed_frames]

    return torch.tensor(mel).unsqueeze(0).float()


# ────────────────────────────────────────────────────────────
# 3. Librosa Baseline
# ────────────────────────────────────────────────────────────
def run_baseline(y):
    """
    直接對原始音訊跑 librosa 的 beat tracker。
    （注意：librosa 不需要 mel tensor，直接吃 raw waveform）

    回傳:
        tempo  : float，估測的 BPM
        beats  : np.array，beat 出現的時間（秒）
    """
    tempo, beat_frames = librosa.beat.beat_track(
        y=y,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        bpm=None,                        # 讓 librosa 自己估
        start_bpm=120                   # 初始猜測值
        
    )

    # beat_frames 是 frame index，換算成秒
    beat_times = librosa.frames_to_time(beat_frames, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)

    return float(tempo), beat_times


# ────────────────────────────────────────────────────────────
# 4. 主程式：把三個模組串在一起
# ────────────────────────────────────────────────────────────
def main():
    api = SoundAPI()

    print("=== Librosa Baseline Pipeline ===")
    print(f"SR={SAMPLE_RATE}, HOP_LENGTH={HOP_LENGTH}, N_MELS={N_MELS}, FIXED_FRAMES={FIXED_FRAMES}")

    loop = 0
    while True:
        loop += 1
        print(f"── Round {loop} ──────────────────────")

        # Step 1: 錄音
        y = api.get_audio()
        print(f"  音訊 shape : {y.shape}")          # 應該是 (66150,)
        print(f"  音訊振幅   : max={y.max():.4f}, mean={np.abs(y).mean():.5f}")

        # 靜音檢查：如果收到的都是接近 0 的訊號，跳過這輪
        if np.abs(y).mean() < 1e-4:
            print("  [靜音，跳過]\n")
            continue

        # Step 2: 轉 Mel Spectrogram（確認 shape 正確）
        mel_tensor = audio_to_mel(y)
        print(f"  Mel shape  : {mel_tensor.shape}")  # 應該是 torch.Size([1, 128, 129])

        # Step 3: Librosa baseline
        tempo, beat_times = run_baseline(y)
        print(f"  BPM        : {tempo:.1f}")
        print(f"  Beat 數量  : {len(beat_times)}")
        print(f"  Beat 時間點: {np.round(beat_times, 2)}")
        print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n停止。")