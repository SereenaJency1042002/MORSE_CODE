import numpy as np
from scipy.signal import butter, filtfilt

class SignalFilter:
    def __init__(self, y, sr):
        self.audio_data = y
        self.sampling_rate = sr

    def filter(self):
        # Step 1: Find tone frequency using FFT
        fft = np.abs(np.fft.rfft(self.audio_data))
        frequencies = np.fft.rfftfreq(len(self.audio_data), 1/self.sampling_rate)
        
        # Look between 200-5000 Hz
        mask = (frequencies >= 200) & (frequencies <= 5000)
        peak_freq = frequencies[mask][np.argmax(fft[mask])]
        
        # Step 2: Adaptive range - use wider band for safety
        low  = max(100, peak_freq - 150)
        high = min(self.sampling_rate / 2 - 100, peak_freq + 150)
        
        # Step 3: Apply bandpass filter
        b, a = butter(5, [low, high], btype='band', fs=self.sampling_rate)
        filtered = filtfilt(b, a, self.audio_data)
        
        # Safety check - if filter failed, return original audio
        if np.isnan(filtered).any() or np.max(np.abs(filtered)) == 0:
            return self.audio_data
        
        return filtered