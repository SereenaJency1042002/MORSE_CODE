import librosa
import numpy as np
import matplotlib.pyplot as plt

def find_morse_frequency(audio_file):
    # Load audio
    y, sr = librosa.load(audio_file)
    
    # Compute FFT
    fft = np.abs(np.fft.rfft(y))
    frequencies = np.fft.rfftfreq(len(y), 1/sr)
    
    # Only look between 200Hz and 3000Hz (Morse range)
    mask = (frequencies >= 200) & (frequencies <= 3000)
    fft_masked = fft[mask]
    freq_masked = frequencies[mask]
    
    # Find the peak frequency
    peak_freq = freq_masked[np.argmax(fft_masked)]
    
    # Plot
    plt.figure(figsize=(12, 4))
    plt.plot(freq_masked, fft_masked)
    plt.axvline(x=peak_freq, color='red', label=f'Peak: {peak_freq:.1f} Hz')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title(f'Frequency Analysis — {audio_file}')
    plt.legend()
    plt.grid()
    plt.show()
    
    print(f"Morse tone detected at: {peak_freq:.1f} Hz")
    return peak_freq

# Test with your files
find_morse_frequency("../audio_files/websdr_recording_start_2026-06-26T08_32_25Z_14589.8kHz.wav")
find_morse_frequency("../audio_files/websdr_recording_start_2026-06-26T08_34_45Z_14589.8kHz.wav")