import sounddevice as sd
import numpy as np
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