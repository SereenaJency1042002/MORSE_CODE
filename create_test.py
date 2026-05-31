import numpy as np
import scipy.io.wavfile as wav

sr = 44100
freq = 700
unit = 0.1  # one unit = 0.1 seconds

def make_beep(units):
    t = np.linspace(0, unit * units, int(sr * unit * units))
    return (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)

def make_silence(units):
    return np.zeros(int(sr * unit * units), dtype=np.int16)

dot     = make_beep(1)      # 1 unit ON
dash    = make_beep(3)      # 3 units ON
sym_gap = make_silence(1)   # gap between dots/dashes
chr_gap = make_silence(3)   # gap between letters
wrd_gap = make_silence(7)   # gap between words

# S = dot dot dot
S = np.concatenate([dot, sym_gap, dot, sym_gap, dot])
# O = dash dash dash
O = np.concatenate([dash, sym_gap, dash, sym_gap, dash])

# SOS with proper gaps
signal = np.concatenate([
    S, chr_gap,
    O, chr_gap,
    S
])

wav.write("audio_files/sos.wav", sr, signal)
print("Created sos.wav successfully!")