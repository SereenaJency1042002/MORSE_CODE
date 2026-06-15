import librosa

class AudioLoader:
    def __init__(self, audio_file):
        self.audio_file = audio_file

    def load(self):
        # Load audio using librosa, y=audio data, sr=sampling rate
        y, sr = librosa.load(self.audio_file)
        return y, sr 