import os
import csv
import random
import numpy as np
import soundfile as sf
import librosa

# ── Configuration ─────────────────────────────────────────────────────────────

SOURCE_FOLDER  = "audio_files/training data set"
SOURCE_CSV     = os.path.join(SOURCE_FOLDER, "labels.csv")
OUTPUT_FOLDER  = "augmented_data"
TRAIN_FOLDER   = os.path.join(OUTPUT_FOLDER, "train")
VAL_FOLDER     = os.path.join(OUTPUT_FOLDER, "val")
SAMPLE_RATE    = 16000
TRAIN_RATIO    = 0.70
RANDOM_SEED    = 42

NOISE_LEVELS = {
    "clean": None,   # no noise
    "low":   20,     # SNR 20dB — light noise
    "med":   10,     # SNR 10dB — medium noise
    "high":  5,      # SNR 5dB  — heavy noise
}


# ── Audio helpers ──────────────────────────────────────────────────────────────

def load_audio(path: str) -> tuple[np.ndarray, int]:
    """Load any audio file (WAV or OGA) and resample to SAMPLE_RATE."""
    y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return y.astype(np.float32), SAMPLE_RATE


def add_noise(audio: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Add white Gaussian noise at a given SNR level.
    SNR_db = 10 * log10(signal_power / noise_power)
    Higher SNR = less noise.
    """
    signal_power = np.mean(audio ** 2)
    if signal_power == 0:
        return audio
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
    return (audio + noise).astype(np.float32)


def save_audio(audio: np.ndarray, path: str) -> None:
    """Save audio array as WAV file."""
    sf.write(path, audio, SAMPLE_RATE, subtype='PCM_16')


# ── CSV helpers ────────────────────────────────────────────────────────────────

def load_labels(csv_path: str) -> list[dict]:
    """Load labels.csv and return list of dicts."""
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def save_labels(rows: list[dict], csv_path: str) -> None:
    """Save rows to labels.csv."""
    if not rows:
        return
    fieldnames = ["filename", "expected_text", "source",
                  "noise_level", "snr_db", "original_file", "split"]
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ── Main augmentation pipeline ─────────────────────────────────────────────────

def augment():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Create output folders
    os.makedirs(TRAIN_FOLDER, exist_ok=True)
    os.makedirs(VAL_FOLDER, exist_ok=True)

    # Load source labels
    labels = load_labels(SOURCE_CSV)
    print(f"Loaded {len(labels)} labeled files from {SOURCE_CSV}")

    # Filter to files that actually exist
    valid_labels = []
    for row in labels:
        src_path = os.path.join(SOURCE_FOLDER, row["filename"])
        if os.path.exists(src_path):
            valid_labels.append(row)
        else:
            print(f"  [SKIP] Not found: {row['filename']}")

    print(f"Valid files: {len(valid_labels)}")

    # Train / val split (on original files, before augmentation)
    random.shuffle(valid_labels)
    split_idx = int(len(valid_labels) * TRAIN_RATIO)
    train_originals = valid_labels[:split_idx]
    val_originals   = valid_labels[split_idx:]

    print(f"Train set: {len(train_originals)} originals "
          f"→ {len(train_originals) * len(NOISE_LEVELS)} augmented")
    print(f"Val set:   {len(val_originals)} originals "
          f"→ {len(val_originals) * len(NOISE_LEVELS)} augmented")
    print("-" * 60)

    train_rows = []
    val_rows   = []

    for split_name, originals, out_folder, rows_list in [
        ("train", train_originals, TRAIN_FOLDER, train_rows),
        ("val",   val_originals,   VAL_FOLDER,   val_rows),
    ]:
        for row in originals:
            src_path = os.path.join(SOURCE_FOLDER, row["filename"])
            base_name = os.path.splitext(row["filename"])[0]

            # Load audio once
            try:
                audio, _ = load_audio(src_path)
            except Exception as e:
                print(f"  [ERROR] Cannot load {row['filename']}: {e}")
                continue

            for noise_label, snr_db in NOISE_LEVELS.items():
                # Apply noise
                if snr_db is None:
                    augmented = audio.copy()
                else:
                    augmented = add_noise(audio, snr_db)

                # Build output filename
                out_filename = f"{base_name}_{noise_label}.wav"
                out_path = os.path.join(out_folder, out_filename)

                # Save
                save_audio(augmented, out_path)

                # Build label row
                rows_list.append({
                    "filename":      out_filename,
                    "expected_text": row["expected_text"],
                    "source":        row.get("source", "generated"),
                    "noise_level":   noise_label,
                    "snr_db":        str(snr_db) if snr_db else "inf",
                    "original_file": row["filename"],
                    "split":         split_name,
                })

                status = "no noise" if snr_db is None else f"SNR {snr_db}dB"
                print(f"  [{split_name.upper()}] {out_filename} ({status})")

    # Save labels
    save_labels(train_rows, os.path.join(TRAIN_FOLDER, "labels.csv"))
    save_labels(val_rows,   os.path.join(VAL_FOLDER,   "labels.csv"))

    print("-" * 60)
    print(f"✅ Train: {len(train_rows)} files → {TRAIN_FOLDER}/")
    print(f"✅ Val:   {len(val_rows)} files → {VAL_FOLDER}/")
    print(f"✅ Total: {len(train_rows) + len(val_rows)} augmented files")
    print(f"✅ Labels saved to train/labels.csv and val/labels.csv")


if __name__ == "__main__":
    augment()
