import os
import pickle
import numpy as np
import librosa
from src.morse_decoder import MORSE_CODE_DICT

MODEL_FILE  = "hmm_model.pkl"
SAMPLE_RATE = 16000

# States
DOT  = 0
DASH = 1


def _load_model() -> dict:
    """Load trained HMM model from disk."""
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(
            f"HMM model not found: {MODEL_FILE}\n"
            f"Run hmm_trainer.py first."
        )
    with open(MODEL_FILE, 'rb') as f:
        return pickle.load(f)


def _log_gaussian_prob(x: float, mean: float, std: float) -> float:
    """Log Gaussian probability density — emission log-probability.

    Computed directly in log-space rather than exp() then log(),
    since exp() underflows to 0 for durations far from the mean
    (e.g. audio at a different WPM than the training data), which
    would otherwise make DOT and DASH indistinguishable.
    """
    std = max(std, 1e-6)
    z = (x - mean) / std
    return -np.log(std) - 0.5 * np.log(2 * np.pi) - 0.5 * z * z


def _viterbi(durations: list, model: dict) -> list:
    """
    Viterbi algorithm — finds most likely state sequence.

    For each pulse duration, decides DOT or DASH based on:
    1. Emission probability (how likely this duration = this state)
    2. Transition probability (what state likely follows)
    3. Prior probability (initial state distribution)

    Returns list of states (DOT=0, DASH=1) for each duration.
    """
    if not durations:
        return []

    n      = len(durations)
    n_states = 2
    trans  = np.array(model["transition_matrix"])
    prior  = model["prior"]

    # Log probabilities for numerical stability
    log_trans = np.log(trans + 1e-10)
    log_prior = np.log(np.array(prior) + 1e-10)

    # Emission log-probabilities
    def log_emit(state, duration):
        if state == DOT:
            return _log_gaussian_prob(duration,
                                      model["dot_mean"],
                                      model["dot_std"])
        else:
            return _log_gaussian_prob(duration,
                                      model["dash_mean"],
                                      model["dash_std"])

    # Viterbi tables
    viterbi_tbl  = np.zeros((n, n_states))
    backpointer  = np.zeros((n, n_states), dtype=int)

    # Initialise first step
    for s in range(n_states):
        viterbi_tbl[0, s] = log_prior[s] + log_emit(s, durations[0])

    # Forward pass
    for t in range(1, n):
        for s in range(n_states):
            scores = [viterbi_tbl[t-1, prev] + log_trans[prev, s]
                      for prev in range(n_states)]
            best_prev           = int(np.argmax(scores))
            viterbi_tbl[t, s]   = scores[best_prev] + log_emit(s, durations[t])
            backpointer[t, s]   = best_prev

    # Backtrack
    states = [0] * n
    states[-1] = int(np.argmax(viterbi_tbl[-1]))
    for t in range(n - 2, -1, -1):
        states[t] = backpointer[t + 1, states[t + 1]]

    return states


def _extract_pulses_with_gaps(audio: np.ndarray, sr: int):
    """
    Extract ON pulse durations AND OFF gap durations.
    Returns list of (is_on, duration_sec, start_frame) tuples.
    """
    frame_length = max(64, int(sr * 0.01))
    hop_length   = frame_length // 2

    rms = librosa.feature.rms(
        y=audio, frame_length=frame_length, hop_length=hop_length
    )[0]

    noise_floor = np.max(rms) * 0.05
    active_rms  = rms[rms > noise_floor]
    if len(active_rms) == 0:
        return []

    threshold = np.median(active_rms) * 0.6
    signal_on = rms > threshold

    segments = []
    current  = signal_on[0]
    count    = 0
    start    = 0
    for i, val in enumerate(signal_on):
        if val == current:
            count += 1
        else:
            dur_sec = (count * hop_length) / sr
            segments.append((bool(current), dur_sec, start))
            start   = i
            current = val
            count   = 1
    if count > 0:
        dur_sec = (count * hop_length) / sr
        segments.append((bool(current), dur_sec, start))

    # Remove noise spikes from ON segments
    on_durs = [d for is_on, d, _ in segments if is_on]
    if not on_durs:
        return []
    min_dur = np.mean(on_durs) * 0.3
    segments = [(io, d, s) for io, d, s in segments
                if not io or d >= min_dur]

    return segments


def _segments_to_text(segments: list, states: list,
                       model: dict) -> tuple[list, str]:
    """
    Convert labelled segments to letters and words.

    Uses dot_mean to derive gap thresholds:
      inter-char gap ≥ 2 × dot_mean
      word gap       ≥ 5 × dot_mean
    """
    # Derive unit from actual audio segments
    # rather than model mean — more robust to speed variation
    on_durs = [d for is_on, d, _ in segments if is_on]
    if on_durs:
        unit = float(np.percentile(on_durs, 30))
    else:
        unit = model["dot_mean"]
    inter_char_thresh = unit * 2.0
    word_gap_thresh   = unit * 5.0

    events       = []
    current_char = []
    state_idx    = 0   # index into states list (only ON segments)

    hop_length = max(64, int(SAMPLE_RATE * 0.01)) // 2

    for is_on, dur_sec, start_frame in segments:
        time_sec = (start_frame * hop_length) / SAMPLE_RATE
        if is_on:
            if state_idx < len(states):
                sym = '.' if states[state_idx] == DOT else '-'
                state_idx += 1
            else:
                sym = '.' if dur_sec < unit * 2.0 else '-'
            current_char.append(sym)
            events.append((time_sec, 'symbol', sym))
        else:
            if dur_sec >= word_gap_thresh:
                if current_char:
                    letter = MORSE_CODE_DICT.get(
                        ''.join(current_char), '?')
                    events.append((time_sec, 'letter', letter))
                    current_char = []
                events.append((time_sec, 'word', ' '))
            elif dur_sec >= inter_char_thresh:
                if current_char:
                    letter = MORSE_CODE_DICT.get(
                        ''.join(current_char), '?')
                    events.append((time_sec, 'letter', letter))
                    current_char = []

    if current_char:
        letter = MORSE_CODE_DICT.get(''.join(current_char), '?')
        events.append((0.0, 'letter', letter))

    text = ''.join(d for _, et, d in events
                   if et in ('letter', 'word'))
    return events, text


class HMMDecoder:
    """
    Morse code decoder using Hidden Markov Model.
    Drop-in replacement for MorseDecoder for batch testing.
    """

    def __init__(self, filtered_audio: np.ndarray,
                 sr: int,
                 morse_code_dict: dict,
                 model_file: str = MODEL_FILE):
        self.audio  = filtered_audio
        self.sr     = sr
        self.model  = _load_model()

    def decode_with_timing(self) -> tuple[list, str]:
        """
        Decode audio using Viterbi HMM.
        Returns (events, text) — same interface as MorseDecoder.
        """
        segments = _extract_pulses_with_gaps(self.audio, self.sr)
        if not segments:
            return [], ""

        on_durations = [d for is_on, d, _ in segments if is_on]
        if not on_durations:
            return [], ""

        states = _viterbi(on_durations, self.model)
        events, text = _segments_to_text(segments, states, self.model)
        return events, text

    def decode(self) -> str:
        _, text = self.decode_with_timing()
        return text
