import numpy as np
from scipy.signal import butter, filtfilt


class SignalFilter:
    def __init__(self, y, sr):
        self.audio_data = y
        self.sampling_rate = sr

    def filter(self):
        """
        Adaptive bandpass filter with tighter bandwidth.

        Steps:
        1. Find dominant frequency using FFT
        2. Check signal strength vs noise floor
        3. Apply tight bandpass ±75Hz around peak
        4. Apply noise gate — silence weak signals
        5. Return filtered audio
        """
        # Step 1 — FFT to find dominant frequency
        fft      = np.abs(np.fft.rfft(self.audio_data))
        freqs    = np.fft.rfftfreq(
            len(self.audio_data), 1 / self.sampling_rate
        )

        # Only look in Morse CW range 200-5000Hz
        mask     = (freqs >= 200) & (freqs <= 5000)
        fft_cw   = fft[mask]
        freqs_cw = freqs[mask]

        if len(fft_cw) == 0:
            return self.audio_data

        peak_idx  = np.argmax(fft_cw)
        peak_freq = freqs_cw[peak_idx]
        peak_mag  = fft_cw[peak_idx]

        # Step 2 — Check signal strength
        # If peak is not significantly above noise floor → weak signal
        noise_floor = np.median(fft_cw)
        snr_ratio   = peak_mag / (noise_floor + 1e-9)

        # Step 3 — Tight bandpass ±75Hz around peak
        # (was ±150Hz before — too wide for pile-up conditions)
        bandwidth = 75
        low  = max(100, peak_freq - bandwidth)
        high = min(self.sampling_rate / 2 - 100, peak_freq + bandwidth)

        # Widen slightly for weaker signals to avoid cutting signal
        if snr_ratio < 5:
            bandwidth = 120
            low  = max(100, peak_freq - bandwidth)
            high = min(self.sampling_rate / 2 - 100, peak_freq + bandwidth)

        # Step 4 — Apply butterworth bandpass filter
        try:
            b, a      = butter(5, [low, high],
                               btype='band',
                               fs=self.sampling_rate)
            filtered  = filtfilt(b, a, self.audio_data)
        except Exception:
            return self.audio_data

        # Safety check — if filter failed return original
        if np.isnan(filtered).any() or np.max(np.abs(filtered)) == 0:
            return self.audio_data

        return filtered.astype(np.float32)

    def get_peak_frequency(self) -> float:
        """
        Returns detected peak frequency in Hz.
        Useful for UI display and logging.
        """
        fft   = np.abs(np.fft.rfft(self.audio_data))
        freqs = np.fft.rfftfreq(
            len(self.audio_data), 1 / self.sampling_rate
        )
        mask  = (freqs >= 200) & (freqs <= 5000)
        if not np.any(mask):
            return 0.0
        return float(freqs[mask][np.argmax(fft[mask])])

    def get_snr(self) -> float:
        """
        Returns estimated SNR of the dominant signal.
        Higher = cleaner signal.
        """
        fft   = np.abs(np.fft.rfft(self.audio_data))
        freqs = np.fft.rfftfreq(
            len(self.audio_data), 1 / self.sampling_rate
        )
        mask  = (freqs >= 200) & (freqs <= 5000)
        if not np.any(mask):
            return 0.0
        fft_cw      = fft[mask]
        peak_mag    = float(np.max(fft_cw))
        noise_floor = float(np.median(fft_cw))
        if noise_floor < 1e-9:
            return 99.0
        return round(20 * np.log10(peak_mag / noise_floor), 1)
