import os
import csv
import json
import time
import numpy as np
from tabulate import tabulate

# ── Configuration ──────────────────────────────────────────────────────────────

VAL_FOLDER  = "augmented_data/val"
VAL_CSV     = os.path.join(VAL_FOLDER, "labels.csv")
REPORT_FILE = "batch_report.json"
SAMPLE_RATE = 16000


# ── Metric helpers ─────────────────────────────────────────────────────────────

def levenshtein(s: str, t: str) -> int:
    m, n = len(s), len(t)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = (prev if s[i-1] == t[j-1]
                     else 1 + min(prev, dp[j], dp[j-1]))
            prev = temp
    return dp[n]


def cer(expected: str, actual: str) -> float:
    if not expected:
        return 0.0
    return levenshtein(expected, actual) / len(expected)


def wer(expected: str, actual: str) -> float:
    exp_words = expected.split()
    act_words = actual.split()
    if not exp_words:
        return 0.0
    return levenshtein(' '.join(exp_words),
                       ' '.join(act_words)) / len(exp_words)


def char_accuracy(expected: str, actual: str) -> float:
    return max(0.0, 1.0 - cer(expected, actual))


def word_accuracy(expected: str, actual: str) -> float:
    return max(0.0, 1.0 - wer(expected, actual))


# ── Decoders ───────────────────────────────────────────────────────────────────

def decode_kmeans(audio_path: str) -> str:
    """Run K-Means decoder on audio file."""
    from src.audio_loader import AudioLoader
    from src.signal_filter import SignalFilter
    from src.morse_decoder import MorseDecoder, MORSE_CODE_DICT
    y, sr = AudioLoader(audio_path).load()
    filtered = SignalFilter(y, sr).filter()
    _, text = MorseDecoder(
        filtered, sr, MORSE_CODE_DICT, mode='kmeans'
    ).decode_with_timing()
    return text.strip().upper()


def decode_hmm(audio_path: str) -> str:
    """Run HMM decoder on audio file."""
    from src.audio_loader import AudioLoader
    from src.signal_filter import SignalFilter
    from src.morse_decoder import MORSE_CODE_DICT
    from src.hmm_decoder import HMMDecoder
    y, sr = AudioLoader(audio_path).load()
    filtered = SignalFilter(y, sr).filter()
    _, text = HMMDecoder(
        filtered, sr, MORSE_CODE_DICT
    ).decode_with_timing()
    return text.strip().upper()


# ── Main batch test ────────────────────────────────────────────────────────────

def run_batch_test():
    rows = []
    with open(VAL_CSV, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    print("Batch Tester — K-Means vs HMM Decoder")
    print(f"Validation set: {len(rows)} files")
    print("=" * 80)

    results      = []
    noise_groups = {}

    for row in rows:
        audio_path = os.path.join(VAL_FOLDER, row["filename"])
        expected   = row["expected_text"].strip().upper()
        noise_lvl  = row.get("noise_level", "unknown")

        if not os.path.exists(audio_path):
            print(f"  [SKIP] {row['filename']}")
            continue

        # K-Means decode
        t0 = time.time()
        try:
            km_decoded = decode_kmeans(audio_path)
        except Exception as e:
            print(f"  [KM ERROR] {row['filename']}: {e}")
            km_decoded = ""
        km_time = time.time() - t0

        # HMM decode
        t0 = time.time()
        try:
            hmm_decoded = decode_hmm(audio_path)
        except Exception as e:
            print(f"  [HMM ERROR] {row['filename']}: {e}")
            hmm_decoded = ""
        hmm_time = time.time() - t0

        # Metrics
        km_c  = char_accuracy(expected, km_decoded)
        km_w  = word_accuracy(expected, km_decoded)
        hmm_c = char_accuracy(expected, hmm_decoded)
        hmm_w = word_accuracy(expected, hmm_decoded)

        result = {
            "filename":       row["filename"],
            "noise_level":    noise_lvl,
            "expected":       expected,
            "km_decoded":     km_decoded,
            "hmm_decoded":    hmm_decoded,
            "km_char_acc":    round(km_c,  4),
            "km_word_acc":    round(km_w,  4),
            "km_cer":         round(cer(expected, km_decoded),  4),
            "km_wer":         round(wer(expected, km_decoded),  4),
            "hmm_char_acc":   round(hmm_c, 4),
            "hmm_word_acc":   round(hmm_w, 4),
            "hmm_cer":        round(cer(expected, hmm_decoded), 4),
            "hmm_wer":        round(wer(expected, hmm_decoded), 4),
            "km_time_sec":    round(km_time,  3),
            "hmm_time_sec":   round(hmm_time, 3),
        }
        results.append(result)

        if noise_lvl not in noise_groups:
            noise_groups[noise_lvl] = []
        noise_groups[noise_lvl].append(result)

        # Winner indicator
        winner = "TIE" if km_c == hmm_c else ("KM" if km_c > hmm_c else "HMM")
        print(f"  [{noise_lvl:5s}] {row['filename'][:35]:35s} "
              f"KM={km_c:.1%} HMM={hmm_c:.1%} → {winner}")

    print("=" * 80)

    # ── Summary table ──────────────────────────────────────────────────────────
    summary_rows = []
    for noise_lvl in ["clean", "low", "med", "high"]:
        group = noise_groups.get(noise_lvl, [])
        if not group:
            continue
        km_avg  = np.mean([r["km_char_acc"]  for r in group])
        hmm_avg = np.mean([r["hmm_char_acc"] for r in group])
        km_wer_avg  = np.mean([r["km_wer"]  for r in group])
        hmm_wer_avg = np.mean([r["hmm_wer"] for r in group])
        improvement = hmm_avg - km_avg
        summary_rows.append([
            noise_lvl.upper(),
            len(group),
            f"{km_avg:.1%}",
            f"{hmm_avg:.1%}",
            f"{improvement:+.1%}",
            f"{km_wer_avg:.3f}",
            f"{hmm_wer_avg:.3f}",
        ])

    # Overall
    if results:
        km_overall  = np.mean([r["km_char_acc"]  for r in results])
        hmm_overall = np.mean([r["hmm_char_acc"] for r in results])
        km_wer_all  = np.mean([r["km_wer"]  for r in results])
        hmm_wer_all = np.mean([r["hmm_wer"] for r in results])
        improvement = hmm_overall - km_overall
        summary_rows.append([
            "OVERALL",
            len(results),
            f"{km_overall:.1%}",
            f"{hmm_overall:.1%}",
            f"{improvement:+.1%}",
            f"{km_wer_all:.3f}",
            f"{hmm_wer_all:.3f}",
        ])

    print("\nSUMMARY — K-Means vs HMM Performance")
    print(tabulate(
        summary_rows,
        headers=["Noise", "Files",
                 "KM Char Acc", "HMM Char Acc", "Improvement",
                 "KM WER", "HMM WER"],
        tablefmt="grid"
    ))

    # ── Save report ────────────────────────────────────────────────────────────
    report = {
        "decoders":    ["kmeans", "hmm"],
        "total_files": len(results),
        "summary": {
            noise: {
                "files":           len(g),
                "km_char_accuracy": round(float(
                    np.mean([r["km_char_acc"] for r in g])), 4),
                "hmm_char_accuracy": round(float(
                    np.mean([r["hmm_char_acc"] for r in g])), 4),
                "improvement": round(float(
                    np.mean([r["hmm_char_acc"] - r["km_char_acc"]
                             for r in g])), 4),
                "km_wer": round(float(
                    np.mean([r["km_wer"] for r in g])), 4),
                "hmm_wer": round(float(
                    np.mean([r["hmm_wer"] for r in g])), 4),
            }
            for noise, g in noise_groups.items()
        },
        "results": results,
    }

    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print(f"\n✅ Report saved to {REPORT_FILE}")


if __name__ == "__main__":
    run_batch_test()
