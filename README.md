# Intelligent Morse Code Decoder

**Object-Oriented Programming Project — TH Köln, Summer Semester 2026**
**Student:** Sereena Jency

A desktop application that takes an audio file containing Morse code and decodes it to English text — using digital signal processing, machine learning, and two layers of intelligent error correction.

---

## What It Does

1. Loads a `.wav` or `.mp3` audio file containing a Morse code transmission
2. Locates the exact tone frequency using FFT and applies a targeted bandpass filter
3. Extracts the signal energy envelope over time using RMS analysis
4. Uses an adaptive threshold to convert the energy curve into a binary ON/OFF signal
5. Applies **K-Means clustering** to learn the dot and dash durations from the audio itself
6. Looks up each dot/dash pattern in the ITU Morse code dictionary
7. Runs a **local correction layer** that fixes unrecognised symbols using Hamming distance and fixes garbled words using Levenshtein edit distance
8. Optionally queries a **Groq LLM (LLaMA 3.3 70B)** for context-aware correction of ham radio patterns, callsigns, and prosigns
9. Displays everything in a warm-themed desktop GUI with an embedded waveform graph

---

## Why I Chose This Project

### 1. Morse Code Is Not a Toy Problem

Morse code is a real, still-active communication protocol. It is used in:
- **Aviation** — pilots still learn it to identify VOR navigation beacons by their Morse identifier
- **Amateur radio** — the international CW (continuous wave) bands are active every day
- **Emergency signalling** — SOS ( `...` `---` `...` ) remains a universal distress signal
- **Assistive technology** — people with motor disabilities use Morse input devices to communicate

Decoding it correctly from real noisy audio is a non-trivial engineering problem. That made it worth building.

### 2. The Problem Naturally Decomposes Into Exactly 6 Classes

The moment you draw the pipeline — load audio → filter → decode → visualise → display → correct — you have your class structure:

| Class | File | Single Responsibility |
|---|---|---|
| `AudioLoader` | `audio_loader.py` | Load any WAV or MP3 |
| `SignalFilter` | `signal_filter.py` | Remove non-Morse frequencies |
| `MorseDecoder` | `morse_decoder.py` | Convert signal to text |
| `SignalVisualizer` | `signal_visualizer.py` | Render the waveform graph |
| `UIDisplay` | `ui_display.py` | Manage the GUI |
| `MorseApp` | `main.py` | Wire all classes together |

This is not a forced OOP structure. The problem itself demanded it. Each class can be understood, tested, and changed independently. Replacing the filter algorithm in `signal_filter.py` does not touch the decoder. Adding a new corrector class did not require changing anything that already worked.

### 3. It Uses Algorithms That Actually Matter

Each algorithm in the pipeline was chosen because it is the right tool for the job:

- **FFT** — the mathematical foundation of all audio and communications engineering
- **Butterworth bandpass filter** — the zero-ripple standard for signal isolation; the reason `filtfilt` is used (not `lfilter`) is timing accuracy
- **K-Means clustering** — the decoder does not assume how fast the sender speaks; it learns dot vs. dash boundaries from the data itself
- **Levenshtein distance** — a classic dynamic programming algorithm that appears in spell checkers, DNA sequencing, and version control diff tools
- **Hamming distance** — used to find the closest Morse pattern when K-Means misclassifies a segment
- **LLM with structured prompting** — used to handle ham radio idioms, callsign protection, and prosign vocabulary that no static dictionary can cover

No algorithm was added for show. Each one solves a specific failure case.

### 4. It Demonstrates That OOP Enables Safe Extension

When the first version was complete (loader + filter + decoder + GUI), adding error correction required **zero changes** to existing code. `IntelligentCorrector` was added as a new class. `AIPredictor` was added as a separate optional class. `UIDisplay` was extended to call them.

This is what the Open/Closed Principle looks like in practice: open for extension, closed for modification. The presentation can show two versions of the output side by side — raw decode vs. corrected — as proof that the architecture absorbed new functionality without breaking what existed.

### 5. It Produces a Demo You Can Actually See

The waveform graph renders the exact signal that the decoder processed. The decoded text appears character by character. The raw output and corrected output are shown together so the correction layer's effect is immediately visible. A live demo is far more compelling than showing unit tests pass.

---

## Pipeline

```
Audio File (.wav / .mp3)
        |
        v
   AudioLoader          librosa.load() → waveform array + sample rate
        |
        v
   SignalFilter         FFT → detect tone frequency → adaptive bandpass → filtfilt
        |
        v
   MorseDecoder         RMS → adaptive threshold → run-length segmentation
                        → K-Means (dot/dash) → K-Means (gap types)
                        → ITU dictionary lookup
        |
        v
   IntelligentCorrector Hamming distance on unrecognised symbols
   (local, offline)     Levenshtein distance on decoded words
        |
        v
   AIPredictor          LLaMA 3.3 70B via Groq API (optional, requires API key)
   (AI, online)         Callsign-aware, prosign-aware, context-aware
        |
        v
   SignalVisualizer     Filtered waveform → matplotlib Figure
        |
        v
   UIDisplay            customtkinter GUI — graph + raw text + corrected text
```

---

## Project Structure

```
morse_project/
├── main.py                     ← MorseApp class — entry point
├── src/
│   ├── audio_loader.py         ← AudioLoader
│   ├── signal_filter.py        ← SignalFilter (FFT + Butterworth)
│   ├── morse_decoder.py        ← MorseDecoder (RMS + K-Means + dictionary)
│   ├── intelligent_corrector.py← IntelligentCorrector (Hamming + Levenshtein)
│   ├── ai_predictor.py         ← AIPredictor (LLaMA 3.3 via Groq API)
│   ├── signal_visualizer.py    ← SignalVisualizer (matplotlib)
│   └── ui_display.py           ← UIDisplay (customtkinter GUI)
├── create_test.py              ← generates a clean SOS test file at 700 Hz
├── pyproject.toml              ← dependencies
├── config.json                 ← configurable parameters
├── main.spec                   ← PyInstaller build spec
├── dist/MorseDecoder/          ← compiled standalone .exe
├── HOW_TO_RUN.md               ← setup guide
└── PROJECT_REPORT.md           ← full technical report
```

---

## Quick Start

### Option A — Run the EXE (no Python required)

1. Open `dist\MorseDecoder\`
2. Double-click `MorseDecoder.exe`

### Option B — Run from Source

**Requirements:** Python 3.11+

```bash
git clone https://github.com/SereenaJency1042002/MORSE_CODE.git
cd MORSE_CODE
python -m venv venv
venv\Scripts\activate        # Windows PowerShell
pip install .
python main.py
```

**To enable the AI correction layer**, create a `.env` file in the project root:

```
GROQ_API_KEY=your_key_here
```

Get a free key at [console.groq.com](https://console.groq.com). If no key is set, the app runs fully offline using only the local corrector.

**To generate a test audio file:**

```bash
python create_test.py
```

This creates a clean SOS signal at 700 Hz in `audio_files/` — a known-correct input for verifying the decoder.

---

## Using the App

1. Click **Load Audio File** and select a `.wav` or `.mp3`
2. Click **Decode**
3. The waveform graph appears and decoding runs
4. The output panel shows the raw decoded text and the corrected text

---

## Libraries

| Library | Role |
|---|---|
| `librosa` | Audio loading and RMS feature extraction |
| `scipy` | Butterworth filter design (`butter`, `filtfilt`) |
| `numpy` | Array operations throughout the pipeline |
| `scikit-learn` | K-Means clustering for dot/dash and gap classification |
| `matplotlib` | Waveform rendering embedded in the GUI |
| `customtkinter` | Modern desktop GUI (themes, rounded widgets) |
| `sounddevice` | Audio playback in the UI |
| `groq` | Groq API client for running LLaMA 3.3 70B correction |

---

## Correction Layers

### Layer 1 — IntelligentCorrector (always active)

- **Symbol correction:** when the decoder outputs `?` for an unrecognised Morse pattern, looks up the closest valid Morse code by Hamming distance (1 edit allowed)
- **Word correction:** compares each decoded word against a word list using Levenshtein distance; replaces words that are 1 edit away from a known word

### Layer 2 — AIPredictor (requires API key)

Sends the corrected text to Meta's LLaMA 3.3 70B with strict rules:
- Never alter callsigns (tokens containing at least one digit embedded in letters)
- Never invent tokens to fill `?` gaps — only fill if context is certain
- Accepts CQ, DE, SK, AR, 73 and other ham radio standard patterns unchanged

Uses `temperature=0.0` for fully deterministic output. Falls back to uncorrected text silently if the API call fails.

### Why Groq + LLaMA 3.3, and not GPT-4o or Claude?

**Speed.** Groq does not make an LLM — it makes custom inference hardware (LPUs) that runs open-source models at extreme speed:

| Provider | Model | Typical speed |
|---|---|---|
| Groq | LLaMA 3.3 70B | ~1000 tokens/second |
| OpenAI | GPT-4o | ~80–120 tokens/second |
| Anthropic | Claude Sonnet | ~100–150 tokens/second |

The correction call happens after decoding while the user is waiting. Groq makes it feel nearly instant. GPT-4o and Claude would introduce a noticeable delay.

**Cost.** Groq has a free tier with limits more than sufficient for a demo. OpenAI and Anthropic are paid with no meaningful free tier — anyone running this project would need a credit card.

**Why not GPT-2?** GPT-2 is a 2019 text *completion* model, not a chat model. It has no concept of system prompts or instruction following. A rule like *"never modify a callsign"* would be ignored — it would just predict the next token. LLaMA 3.3 is instruction-tuned and reliably follows structured rules.

**Why not a local model (Ollama etc.)?** Running LLaMA 70B locally requires a GPU with ~40 GB VRAM. On a laptop CPU it takes 30–60 seconds per response. Groq delivers the same model quality at API speed with no local hardware requirement.

**Quality.** For short, rule-constrained correction tasks (under 250 tokens in, under 50 tokens out), LLaMA 3.3 70B performs comparably to GPT-4o and Claude Sonnet. The task is not open-ended reasoning — it is narrow and structured. A 70B open-source model is more than sufficient.

---

## Why These Approaches Were Chosen

### FFT to detect the tone frequency

The alternative is hardcoding a filter range (e.g. always filter 600–1200 Hz). That fails when a recording uses a tone outside the assumed range. FFT analyses the whole recording and finds exactly where the energy is concentrated — the dominant peak between 200–5000 Hz is always the Morse tone. The filter is then built around that exact frequency, making the system work on any recording regardless of what tone was used.

### Butterworth filter, not Chebyshev or Elliptic

Butterworth is "maximally flat in the passband" — it produces zero ripple inside the frequency range being kept. Chebyshev and Elliptic filters cut off more sharply but introduce oscillations inside the passband. Those oscillations distort the amplitude of the tone, which corrupts the RMS energy curve and makes ON/OFF detection unreliable. For Morse decoding, a clean signal shape matters more than a sharp cutoff edge.

### `filtfilt` instead of `lfilter`

`lfilter` applies the filter once, forward in time, which introduces phase delay — the filtered signal is shifted slightly in time relative to the original. The decoder measures how long each ON segment lasts. If every segment is shifted, duration measurements are wrong, and dots may be misclassified as dashes. `filtfilt` applies the filter forward then backward; the two phase shifts cancel out exactly. Every tone boundary is at the correct timestamp.

### RMS envelope, not raw amplitude

Raw amplitude oscillates at the tone frequency (700 oscillations per second for a 700 Hz tone). A single sample tells you where in the wave cycle you are, not whether the tone is on or off. RMS averages the energy over a short window (10 ms), producing a smooth curve that is high when a tone plays and near zero during silence. The 10 ms window was chosen because Morse dots at typical speed are 60–120 ms — short enough to catch the edges of each symbol, long enough to smooth out wave oscillations.

### Adaptive threshold, not a fixed one

Different recordings have different overall loudness. A fixed threshold works on one file and fails on another. The adaptive threshold is derived from each recording:
- `noise_floor = max_RMS × 0.05` — ignores the quietest 5% as background noise
- `threshold = median(active_RMS) × 0.6` — sets the boundary at 60% of the typical signal energy

The median is used instead of the mean because loud noise spikes would pull the mean upward, making the threshold too high and causing weak tones to be missed.

### K-Means for dot vs. dash classification

The alternative is a hardcoded time threshold (e.g. "shorter than 100 ms = dot"). That fails because sender speed varies between recordings — a fast operator sends 50 ms dots, a slow one sends 200 ms dots. A fixed threshold misclassifies on any recording that differs from the assumed speed.

K-Means groups all ON segment durations into 2 clusters and finds the natural boundary between them from the data itself. The decoder does not assume anything about sender speed — it learns it from each specific file. `n_clusters=2` reflects the fact that there are exactly two types of ON segments in Morse code: dots and dashes.

### Hamming distance for symbol correction

When K-Means misclassifies a single dot as a dash, the resulting Morse pattern does not match any dictionary entry and the decoder outputs `?`. Hamming distance counts how many positions differ between two equal-length strings — fast, simple, and correct for this type of error. A one-dot misclassification produces a Hamming distance of 1. The corrector scans the dictionary for the closest entry and substitutes it, but only if the distance is exactly 1 (never on distance 2+ to avoid wrong guesses).

Levenshtein is not used here because Morse symbol errors are substitutions (a dot becomes a dash), not insertions or deletions. Hamming is the right tool at the symbol level.

### Levenshtein distance for word correction

After symbol correction, a decoded word might still be `HELO` (missing letter) or `WROLD` (wrong letter order). Levenshtein distance handles all three error types: substitution, insertion, and deletion. A distance of 1 means exactly one thing went wrong. The corrector only replaces a word when the closest known word is 1 edit away — confident enough to correct, strict enough to avoid wrong guesses.

Hamming cannot be used here because it only compares equal-length strings. `HELO` and `HELLO` have different lengths, so Hamming cannot measure the distance between them.

### Summary table

| Decision | Why this, not the alternative |
|---|---|
| FFT to find tone | Adapts to any recording; hardcoding a range misses tones outside the assumed window |
| Butterworth filter | Zero ripple in the passband — preserves signal shape, which is critical for timing |
| `filtfilt` | Zero phase shift — phase delay from `lfilter` corrupts dot/dash duration measurements |
| RMS envelope | Raw amplitude oscillates too fast; RMS gives the smooth energy curve needed for ON/OFF detection |
| Adaptive threshold | Scales with each recording's loudness; a fixed threshold fails on quiet or loud files |
| K-Means (dot/dash) | Learns sender speed from the data; a hardcoded time split fails on recordings with different speeds |
| Hamming distance | Correct tool for single substitution errors in equal-length Morse symbol sequences |
| Levenshtein distance | Handles insertions and deletions that Hamming cannot; captures missing-letter errors |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Activate the venv and run `pip install .` |
| PowerShell blocks `Activate.ps1` | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| EXE flagged by antivirus | False positive common with PyInstaller — add an exception |
| Decoding gives wrong output | Run `create_test.py` first and test with the generated SOS file |
| AI correction does nothing | Check that `GROQ_API_KEY` is set in `.env` and the key is valid |

---

## Known Limitations

- **Word gap detection on noisy audio:** K-Means clusters gap durations well when the 1:3:7 Morse timing ratio is preserved. Real audio often compresses these ratios, causing letter gaps to be misclassified as word gaps and vice versa.
- **Very slow or very fast transmission:** The adaptive threshold and K-Means handle a wide speed range, but extremely slow transmissions (long silences relative to dots) may confuse the gap classifier.
- **MP3 artefacts:** MP3 compression introduces spectral noise; results are best with WAV files.

---

> Full technical explanation including line-by-line documentation of every algorithm is in `PROJECT_REPORT.md`.
