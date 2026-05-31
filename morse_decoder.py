import numpy as np
import librosa
from sklearn.cluster import KMeans

MORSE_CODE_DICT = {
    '.-': 'A',    '-...': 'B',  '-.-.': 'C',
    '-..': 'D',   '.': 'E',     '..-.': 'F',
    '--.': 'G',   '....': 'H',  '..': 'I',
    '.---': 'J',  '-.-': 'K',   '.-..': 'L',
    '--': 'M',    '-.': 'N',    '---': 'O',
    '.--.': 'P',  '--.-': 'Q',  '.-.': 'R',
    '...': 'S',   '-': 'T',     '..-': 'U',
    '...-': 'V',  '.--': 'W',   '-..-': 'X',
    '-.--': 'Y',  '--..': 'Z',
    '-----': '0', '.----': '1', '..---': '2',
    '...--': '3', '....-': '4', '.....': '5',
    '-....': '6', '--...': '7', '---..': '8',
    '----.': '9'
}

class MorseDecoder:
    def __init__(self, filtered_audio, sr, morse_code_dict):
        self.filtered_audio = filtered_audio
        self.sr = sr
        self.morse_code_dict = morse_code_dict

    def decode(self):
        S, _ = librosa.magphase(librosa.stft(self.filtered_audio))
        rms = librosa.feature.rms(S=S)[0]
        threshold = 0.5 * np.max(rms)
        signal_on = rms > threshold

        segments = []
        current = signal_on[0]
        count = 0
        for val in signal_on:
            if val == current:
                count += 1
            else:
                segments.append((bool(current), count))
                current = val
                count = 1
        segments.append((bool(current), count))

        on_durations = np.array([d for s, d in segments if s]).reshape(-1, 1)
        off_durations = np.array([d for s, d in segments if not s]).reshape(-1, 1)

        if len(on_durations) < 2:
            return ""

        km_on = KMeans(n_clusters=2, n_init=10, random_state=0)
        km_on.fit(on_durations)
        dot_cluster = int(np.argmin(km_on.cluster_centers_))

        n_off_clusters = min(3, len(off_durations))
        km_off = KMeans(n_clusters=n_off_clusters, n_init=10, random_state=0)
        km_off.fit(off_durations)
        sorted_centers = np.sort(km_off.cluster_centers_.flatten())

        morse_sequence = []
        current_char = []

        for is_on, duration in segments:
            if is_on:
                label = km_on.predict([[duration]])[0]
                current_char.append('.' if label == dot_cluster else '-')
            else:
                label = km_off.predict([[duration]])[0]
                center = km_off.cluster_centers_[label][0]
                if n_off_clusters == 3:
                    if center == sorted_centers[2]:
                        if current_char:
                            morse_sequence.append(''.join(current_char))
                            current_char = []
                        morse_sequence.append(' ')
                    elif center == sorted_centers[1]:
                        if current_char:
                            morse_sequence.append(''.join(current_char))
                            current_char = []
                else:
                    if center == sorted_centers[-1]:
                        if current_char:
                            morse_sequence.append(''.join(current_char))
                            current_char = []

        if current_char:
            morse_sequence.append(''.join(current_char))

        result = []
        for code in morse_sequence:
            if code == ' ':
                result.append(' ')
            else:
                result.append(self.morse_code_dict.get(code, '?'))

        return ''.join(result)