import os
import csv
import pickle
import numpy as np
import librosa
from scipy.signal import butter, filtfilt
from src.morse_decoder import MORSE_CODE_DICT

# ── Configuration ──────────────────────────────────────────────────────────────

TRAIN_FOLDER = "augmented_data/train"
TRAIN_CSV    = os.path.join(TRAIN_FOLDER, "labels.csv")
MODEL_FILE   = "hmm_model.pkl"
SAMPLE_RATE  = 16000

# HMM states
DOT  = 0
DASH = 1

# Inverted Morse dict: letter → pattern string
LETTER_TO_MORSE = {v: k for k, v in MORSE_CODE_DICT.items()
                   if len(v) == 1}  # single characters only


# ── Audio processing ───────────────────────────────────────────────────────────

def load_and_filter(path: str):
    """Load audio and apply bandpass filter."""
    y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    y = y.astype(np.float32)

    # Find dominant frequency
    fft  = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), 1 / sr)
    mask  = (freqs >= 200) & (freqs <= 5000)
    peak  = freqs[mask][np.argmax(fft[mask])]

    # Bandpass filter
    low  = max(100, peak - 150)
    high = min(sr / 2 - 100, peak + 150)
    b, a = butter(5, [low, high], btype='band', fs=sr)
    filtered = filtfilt(b, a, y)

    if np.isnan(filtered).any() or np.max(np.abs(filtered)) == 0:
        return y, sr
    return filtered.astype(np.float32), sr


def extract_pulses(audio: np.ndarray, sr: int):
    """
    Extract ON pulse durations from audio.
    Returns list of durations in seconds.
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

    threshold  = np.median(active_rms) * 0.6
    signal_on  = rms > threshold

    # Find ON segments
    durations = []
    count = 0
    for val in signal_on:
        if val:
            count += 1
        else:
            if count > 0:
                dur_sec = (count * hop_length) / sr
                durations.append(dur_sec)
                count = 0
    if count > 0:
        durations.append((count * hop_length) / sr)

    # Remove noise spikes
    if not durations:
        return []
    mean_dur = np.mean(durations)
    durations = [d for d in durations if d >= mean_dur * 0.3]

    return durations


def label_pulses(durations: list, expected_text: str):
    """
    Label each pulse duration as DOT or DASH
    using the known expected text.

    Strategy:
    - Convert expected text to sequence of dots/dashes
    - Match pulse count to symbol count
    - Label each pulse accordingly
    """
    # Build expected dot/dash sequence from text
    symbols = []
    for char in expected_text.replace(' ', ''):
        char = char.upper()
        if char in LETTER_TO_MORSE:
            for sym in LETTER_TO_MORSE[char]:
                symbols.append(DOT if sym == '.' else DASH)

    if not durations or not symbols:
        return [], []

    # If counts match — direct labeling
    if len(durations) == len(symbols):
        return durations, symbols

    # If counts don't match — use duration ratio to label
    # (fallback: shorter = dot, longer = dash)
    if len(durations) >= 2:
        median_dur = np.median(durations)
        labels = [DOT if d < median_dur * 1.8 else DASH
                  for d in durations]
        return durations, labels

    return [], []


# ── HMM parameter estimation ───────────────────────────────────────────────────

def train_hmm(all_durations: list, all_labels: list) -> dict:
    """
    Estimate HMM parameters from labeled pulse data.

    States: DOT=0, DASH=1
    Returns dict with:
      - transition_matrix: 2×2 array
      - dot_mean, dot_std: Gaussian emission for DOT state
      - dash_mean, dash_std: Gaussian emission for DASH state
      - prior: [P(start=DOT), P(start=DASH)]
    """
    dot_durations  = [d for d, l in zip(all_durations, all_labels) if l == DOT]
    dash_durations = [d for d, l in zip(all_durations, all_labels) if l == DASH]

    print(f"  DOT  samples: {len(dot_durations)}")
    print(f"  DASH samples: {len(dash_durations)}")

    # Emission parameters (Gaussian)
    dot_mean  = float(np.mean(dot_durations))  if dot_durations  else 0.05
    dot_std   = float(np.std(dot_durations))   if dot_durations  else 0.01
    dash_mean = float(np.mean(dash_durations)) if dash_durations else 0.15
    dash_std  = float(np.std(dash_durations))  if dash_durations else 0.03

    # Transition matrix from label sequences
    trans = np.ones((2, 2))  # Laplace smoothing
    for i in range(len(all_labels) - 1):
        trans[all_labels[i]][all_labels[i + 1]] += 1
    trans = trans / trans.sum(axis=1, keepdims=True)

    # Prior (initial state probabilities)
    n_dot  = len(dot_durations)
    n_dash = len(dash_durations)
    total  = n_dot + n_dash
    prior  = [n_dot / total, n_dash / total] if total > 0 else [0.5, 0.5]

    return {
        "transition_matrix": trans.tolist(),
        "dot_mean":   dot_mean,
        "dot_std":    max(dot_std, 0.001),
        "dash_mean":  dash_mean,
        "dash_std":   max(dash_std, 0.001),
        "prior":      prior,
        "n_samples":  total,
    }


# ── Main training pipeline ─────────────────────────────────────────────────────

def run_training():
    print("HMM Trainer — Morse Code Decoder")
    print("=" * 60)

    # Load training labels
    rows = []
    with open(TRAIN_CSV, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    # Only use clean files for training
    # (avoids noise affecting timing statistics)
    clean_rows = [r for r in rows if r.get("noise_level") == "clean"]
    print(f"Training on {len(clean_rows)} clean files "
          f"(from {len(rows)} total)")
    print("-" * 60)

    all_durations = []
    all_labels    = []
    skipped       = 0

    for row in clean_rows:
        audio_path    = os.path.join(TRAIN_FOLDER, row["filename"])
        expected_text = row["expected_text"].strip().upper()

        if not os.path.exists(audio_path):
            print(f"  [SKIP] {row['filename']}")
            skipped += 1
            continue

        try:
            audio, sr = load_and_filter(audio_path)
            durations = extract_pulses(audio, sr)
            durs, labs = label_pulses(durations, expected_text)

            if durs:
                all_durations.extend(durs)
                all_labels.extend(labs)
                print(f"  ✅ {row['filename'][:50]:50s} "
                      f"→ {len(durs)} pulses")
            else:
                print(f"  ⚠️  {row['filename'][:50]:50s} "
                      f"→ no pulses extracted")
                skipped += 1

        except Exception as e:
            print(f"  ❌ {row['filename']}: {e}")
            skipped += 1

    print("-" * 60)
    print(f"Total pulses collected: {len(all_durations)}")
    print(f"Files skipped: {skipped}")

    if len(all_durations) < 10:
        print("❌ Not enough data to train HMM. Need at least 10 pulses.")
        return

    # Train HMM
    print("\nEstimating HMM parameters...")
    model = train_hmm(all_durations, all_labels)

    # Print learned parameters
    print(f"\nLearned parameters:")
    print(f"  DOT  → mean={model['dot_mean']*1000:.1f}ms  "
          f"std={model['dot_std']*1000:.1f}ms")
    print(f"  DASH → mean={model['dash_mean']*1000:.1f}ms  "
          f"std={model['dash_std']*1000:.1f}ms")
    print(f"  Ratio DASH/DOT = "
          f"{model['dash_mean']/model['dot_mean']:.2f}x "
          f"(ideal = 3.0x)")
    print(f"  Prior: DOT={model['prior'][0]:.2f} "
          f"DASH={model['prior'][1]:.2f}")
    print(f"  Transition matrix:")
    tm = model['transition_matrix']
    print(f"    DOT  → DOT={tm[0][0]:.2f}  DASH={tm[0][1]:.2f}")
    print(f"    DASH → DOT={tm[1][0]:.2f}  DASH={tm[1][1]:.2f}")

    # Save model
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)

    print(f"\n✅ Model saved to {MODEL_FILE}")
    print(f"   Trained on {model['n_samples']} pulse samples")


if __name__ == "__main__":
    run_training()
