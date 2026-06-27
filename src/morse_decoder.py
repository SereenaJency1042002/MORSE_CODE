import numpy as np
import librosa
from sklearn.cluster import KMeans

MORSE_CODE_DICT = {
    # Letters
    '.-':   'A', '-...': 'B', '-.-.': 'C',
    '-..':  'D', '.':    'E', '..-.': 'F',
    '--.':  'G', '....': 'H', '..':   'I',
    '.---': 'J', '-.-':  'K', '.-..': 'L',
    '--':   'M', '-.':   'N', '---':  'O',
    '.--.': 'P', '--.-': 'Q', '.-.':  'R',
    '...':  'S', '-':    'T', '..-':  'U',
    '...-': 'V', '.--':  'W', '-..-': 'X',
    '-.--': 'Y', '--..': 'Z',

    # Numbers
    '-----': '0', '.----': '1', '..---': '2',
    '...--': '3', '....-': '4', '.....': '5',
    '-....': '6', '--...': '7', '---..': '8',
    '----.': '9',

    # Punctuation
    '.-.-.-': '.',
    '--..--': ',',
    '..--..': '?',
    '-..-.':  '/',
    '-...-':  '=',
    '-.--.' : '(',
    '-.--.-': ')',
    '.-...':  '&',
    '---...': ':',
    '-.-.-.': ';',
    '.-.-.':  '+',
    '-....-': '-',
    '..--.-': '_',
    '.-..-.': '"',
    '...-..-': '$',
    '.--.-.' : '@',

    # Prosigns (sent as joined — no gap between letters)
    '.-.-':   'AA',  # New line
    '-.-..-': 'CL',  # Clear — closing station
    # '-.-.': 'CT' omitted — duplicates 'C' key, causes every C to decode as CT
    '...-.':  'SN',  # Understood
    '...-.-': 'SK',  # End of contact
    # '.-.-.': 'AR' omitted — duplicates '+' key
    '-...-.-':'BK',  # Break
    '......': 'HH',  # Error — disregard
}

# Word list for intelligent correction
WORD_LIST = [
    # Common English
    'HELLO', 'WORLD', 'YES', 'NO', 'THE', 'AND', 'FOR',
    'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER',
    'WAS', 'ONE', 'OUR', 'OUT', 'DAY', 'GET', 'HAS',
    'HIM', 'HIS', 'HOW', 'ITS', 'MAY', 'NEW', 'NOW',
    'OLD', 'SEE', 'TWO', 'WAY', 'WHO', 'BOY', 'DID',
    'MAN', 'END', 'PUT', 'SAY', 'SHE', 'TOO', 'USE',

    # Amateur radio prosigns
    'CQ', 'DE', 'AR', 'SK', 'BK', 'KN', 'QSO',
    'QTH', 'QRM', 'QRN', 'QRZ', 'QSB', 'QRP', 'QRO',
    'QRX', 'QRT', 'QSL', 'QSY', 'QTR',

    # Common ham radio words
    'TNX', 'TKS', 'PSE', 'PLS', 'RST', 'RIG', 'ANT',
    'PWR', 'OP', 'NAME', 'NR', 'NW', 'OM', 'YL', 'XYL',
    'FB', 'OK', 'ROGER', 'COPY', 'OVER', 'OUT',
    'SIGNAL', 'RADIO', 'STATION', 'BAND', 'FREQ',
    'WATTS', 'METER', 'ANTENNA', 'REPORT',

    # Numbers as words
    'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE',
    'SIX', 'SEVEN', 'EIGHT', 'NINE', 'ZERO',
]


class MorseDecoder:
    def __init__(self, filtered_audio, sr, morse_code_dict):
        self.filtered_audio = filtered_audio
        self.sr = sr
        self.morse_code_dict = morse_code_dict

    def decode_with_timing(self):
        frame_length = max(64, int(self.sr * 0.01))
        hop_length = frame_length // 2
        rms = librosa.feature.rms(
            y=self.filtered_audio,
            frame_length=frame_length,
            hop_length=hop_length
        )[0]

        noise_floor = np.max(rms) * 0.05
        active_rms = rms[rms > noise_floor]
        threshold = np.median(active_rms) * 0.6 if len(active_rms) > 0 else np.max(rms) * 0.5
        signal_on = rms > threshold

        segments = []
        current = signal_on[0]
        count = 0
        start = 0
        for i, val in enumerate(signal_on):
            if val == current:
                count += 1
            else:
                segments.append((bool(current), count, start))
                start = i
                current = val
                count = 1
        segments.append((bool(current), count, start))

        on_durations = np.array([d for s, d, _ in segments if s]).reshape(-1, 1)

        # Filter out noise spikes
        min_duration = np.mean(on_durations) * 0.3
        segments = [(s, d, st) for s, d, st in segments if not s or d >= min_duration]

        on_durations = np.array([d for s, d, _ in segments if s]).reshape(-1, 1)
        off_durations = np.array([d for s, d, _ in segments if not s]).reshape(-1, 1)

        if len(on_durations) < 2:
            return [], ""

        km_on = KMeans(n_clusters=2, n_init=10, random_state=0)
        km_on.fit(on_durations)
        dot_cluster = int(np.argmin(km_on.cluster_centers_))
        dash_cluster = 1 - dot_cluster

        centers = km_on.cluster_centers_.flatten()
        ratio = centers[dash_cluster] / (centers[dot_cluster] + 1e-9)
        if ratio < 2.0:
            _on_split = float(np.mean(on_durations))
            _use_kmeans_on = False
        else:
            _on_split = None
            _use_kmeans_on = True

        n_off_clusters = min(3, len(off_durations))
        km_off = KMeans(n_clusters=n_off_clusters, n_init=10, random_state=0)
        km_off.fit(off_durations)
        sorted_centers = np.sort(km_off.cluster_centers_.flatten())

        # Word gap threshold based on Morse timing rule
        word_gap_threshold = (sorted_centers[1] + sorted_centers[2]) / 2 if n_off_clusters == 3 else None

        events = []
        current_char = []

        for is_on, duration, start_frame in segments:
            time_sec = (start_frame * hop_length) / self.sr
            if is_on:
                if _use_kmeans_on:
                    label = km_on.predict([[duration]])[0]
                    symbol = '.' if label == dot_cluster else '-'
                else:
                    symbol = '.' if duration < _on_split else '-'
                current_char.append(symbol)
                events.append((time_sec, 'symbol', symbol))
            else:
                label = km_off.predict([[duration]])[0]
                center = km_off.cluster_centers_[label][0]
                if n_off_clusters == 3:
                    if duration > word_gap_threshold:
                        if current_char:
                            letter = self.morse_code_dict.get(''.join(current_char), '?')
                            events.append((time_sec, 'letter', letter))
                            current_char = []
                        events.append((time_sec, 'word', ' '))
                    elif center == sorted_centers[1]:
                        if current_char:
                            letter = self.morse_code_dict.get(''.join(current_char), '?')
                            events.append((time_sec, 'letter', letter))
                            current_char = []
                else:
                    if center == sorted_centers[-1]:
                        if current_char:
                            letter = self.morse_code_dict.get(''.join(current_char), '?')
                            events.append((time_sec, 'letter', letter))
                            current_char = []

        if current_char:
            letter = self.morse_code_dict.get(''.join(current_char), '?')
            end_time = len(rms) * hop_length / self.sr
            events.append((end_time, 'letter', letter))

        full_text = ''.join(d for _, et, d in events if et in ('letter', 'word'))
        return events, full_text

    def decode(self):
        _, text = self.decode_with_timing()
        return text