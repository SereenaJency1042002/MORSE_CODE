from scipy.signal import butter, filtfilt

class SignalFilter:
    def __init__(self, y, sr):
        self.audio_data = y
        self.sampling_rate = sr

    def filter(self):
        low = 600
        high = 1200
        b, a = butter(5, [low, high], btype='band', fs=self.sampling_rate)
        filtered = filtfilt(b, a, self.audio_data)
        #filtered = filtered / np.max(np.abs(filtered))  
        return filtered