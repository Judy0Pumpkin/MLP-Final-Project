import sounddevice as sd
import numpy as np

class SoundAPI:
    def __init__(self, sr=22050, duration=1.0):
        self.sr = sr
        self.duration = duration

    def get_audio(self):
        audio = sd.rec(int(self.sr * self.duration),
                       samplerate=self.sr,
                       channels=1)

        sd.wait()

        return audio.flatten()