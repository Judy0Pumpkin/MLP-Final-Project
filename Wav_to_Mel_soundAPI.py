import numpy as np
import librosa
import torch

from constants import FIXED_FRAMES, SAMPLE_RATE, N_MELS, HOP_LENGTH, N_FFT, F_MIN, F_MAX, FIXED_FRAMES

def audio_to_mel(y, sr=SAMPLE_RATE, n_mels=N_MELS, fixed_frames=FIXED_FRAMES):

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels) #把原始音訊轉成Mel spectrogram
    mel = librosa.power_to_db(mel, ref=np.max) #分貝db
    mel = (mel - mel.mean()) / (mel.std() + 1e-6) #標準化

    if mel.shape[1] < fixed_frames: #如果資料時間長度太短需補0(padding)
        mel = np.pad(mel, ((0,0),(0,fixed_frames - mel.shape[1])), mode="constant")
    else:
        mel = mel[:, :fixed_frames]

    return torch.tensor(mel).unsqueeze(0).float()
