# Intelligent Morse Code Decoder
### Object-Oriented Programming — TH Köln, Summer Semester 2026
**Student:** Sereena Jency | **Presentation 1 Deadline:** 29 June 2026

---

## Table of Contents

1. [What the Project Does](#1-what-the-project-does)
2. [System Pipeline](#2-system-pipeline)
3. [OOP Design — Why 6 Separate Classes](#3-oop-design--why-6-separate-classes)
4. [Class-by-Class Technical Breakdown](#4-class-by-class-technical-breakdown)
   - [AudioLoader](#audioloader)
   - [SignalFilter](#signalfilter)
   - [MorseDecoder](#morsedecoder)
   - [SignalVisualizer](#signalvisualizer)
   - [UIDisplay](#uidisplay)
5. [K-Means Clustering — Full Explanation](#5-k-means-clustering--full-explanation)
6. [Libraries Used and Why](#6-libraries-used-and-why)
7. [Build and Distribution](#7-build-and-distribution)
8. [Known Limitation — Word Gap Detection](#8-known-limitation--word-gap-detection)
9. [Planned Improvement — MorseCorrector](#9-planned-improvement--morsecorrector)
10. [Professor's Suggestion — AI Word Gap Prediction](#10-professors-suggestion--ai-word-gap-prediction)
11. [Comparison — Both Approaches](#11-comparison--both-approaches)
12. [Combined Ideal Architecture](#12-combined-ideal-architecture)

---

## 1. What the Project Does

A desktop application that:
- Loads an audio file (WAV or MP3) containing a Morse code signal
- Filters the audio to isolate the Morse tone frequency range
- Processes the signal to detect ON/OFF tone segments
- Uses K-Means machine learning to classify those segments as dots, dashes, and gaps
- Decodes the result into readable English text
- Displays a waveform graph and animates the decoded symbols in sync with audio playback

---

## 2. System Pipeline

```
Audio File (WAV / MP3)
        |
        v
   AudioLoader
   - librosa.load()
   - Returns: raw signal array y[], sample rate sr
        |
        v
   SignalFilter
   - Butterworth bandpass filter (600 Hz – 1200 Hz)
   - Returns: filtered signal (noise removed)
        |
        v
   MorseDecoder
   - RMS envelope extraction (energy per frame)
   - Adaptive threshold → binary signal_on[] array
   - Run-length segmentation → list of (ON/OFF, duration, timestamp)
   - K-Means on ON durations → dot or dash
   - K-Means on OFF durations → intra-letter / letter / word gap
   - Dictionary lookup → decoded text
   - Returns: events[] with timestamps, full decoded string
        |
        v
   SignalVisualizer
   - matplotlib figure of filtered waveform
   - Returns: Figure object (embedded in GUI)
        |
        v
   UIDisplay
   - customtkinter window
   - Embedded waveform graph + animated playhead
   - Animated dot/dash/letter display synced to audio
   - Decoded text output box
```

**Why this order?**
You cannot decode what you have not cleaned. Raw audio contains noise across many frequencies. Filtering must come before decoding. Visualization uses the filtered signal (not raw) because it shows the Morse pattern clearly. The UI depends on all other classes being ready.

---

## 3. OOP Design — Why 6 Separate Classes

Each class has exactly one responsibility. This follows the **Single Responsibility Principle**, the core rule of Object-Oriented Programming.

| Class | File | Single Responsibility |
|---|---|---|
| `AudioLoader` | `audio_loader.py` | Load the audio file |
| `SignalFilter` | `signal_filter.py` | Clean the signal |
| `MorseDecoder` | `morse_decoder.py` | Interpret the signal into text |
| `SignalVisualizer` | `signal_visualizer.py` | Draw the waveform |
| `UIDisplay` | `ui_display.py` | Manage the GUI |
| `MorseApp` | `main.py` | Entry point — wires all classes together |

**Why separate them?**
If all logic were in one file, changing the filter algorithm would risk breaking the UI. Changing the GUI would risk breaking the decoder. By separating them, each class can be tested, changed, and understood independently. For example: the filter frequency range (600–1200 Hz) can be adjusted in `signal_filter.py` without touching any other file.

---

## 4. Class-by-Class Technical Breakdown

### AudioLoader

```python
class AudioLoader:
    def load(self):
        y, sr = librosa.load(self.audio_file)
        return y, sr
```

**What it does:**
Loads the audio file and returns a NumPy float32 array `y[]` (the signal samples) and `sr` (sample rate, default 22,050 Hz).

**Why `librosa.load()`?**
librosa is the standard Python audio processing library. One call handles both WAV and MP3, automatically converts stereo to mono, and resamples to a consistent 22,050 Hz. Doing this manually would require format-specific code for each file type.

**Why 22,050 Hz sample rate?**
The Nyquist theorem states you need a sample rate at least twice the highest frequency you want to capture. Morse tones sit at 600–1000 Hz. At 22,050 Hz, you can accurately represent frequencies up to 11,025 Hz — far more than needed. No information about the Morse tone is lost.

---

### SignalFilter

```python
class SignalFilter:
    def filter(self):
        b, a = butter(5, [600, 1200], btype='band', fs=self.sampling_rate)
        filtered = filtfilt(b, a, self.audio_data)
        return filtered
```

**What it does:**
Applies a Butterworth bandpass filter that keeps only frequencies between 600 Hz and 1200 Hz and removes everything else.

**Why a bandpass filter?**
A Morse audio file contains the actual tone (e.g., 700 Hz) plus background noise at many other frequencies. A bandpass filter lets through only the range where Morse tones exist. Without this, noise at other frequencies would appear as false ON/OFF segments and break the decoder.

**Why 600–1200 Hz?**
Standard Morse practice tones sit in this range, typically 700–900 Hz. The range is wide enough to catch variations across different recordings but narrow enough to block voice, hum, and high-frequency noise.

**Why Butterworth filter?**
Butterworth is called a "maximally flat" filter — it has no ripple inside the passband, meaning it does not distort the signal it is supposed to keep. Other filters (Chebyshev, elliptic) cut more sharply but introduce oscillations. For Morse code, a clean response inside the passband matters more than an extremely sharp cutoff edge.

**Why order 5?**
Filter order controls how steeply the filter cuts off. Order 5 gives a steep enough roll-off to strongly reject noise outside 600–1200 Hz without excessive computational cost or instability.

**Why `filtfilt()` instead of `lfilter()`?**
`lfilter()` applies the filter in one direction, which introduces phase delay — the signal is shifted in time. This means tone boundaries (where ON becomes OFF) would be measured at the wrong timestamps, and dot/dash durations would be wrong. `filtfilt()` applies the filter forward then backward, which cancels the phase shift completely. Accurate timing is critical for correct Morse decoding.

---

### MorseDecoder

This is the core class. It contains all the signal analysis and machine learning logic.

#### Step 1: RMS Envelope Extraction

```python
frame_length = max(64, int(sr * 0.01))   # ~10ms per frame
hop_length   = frame_length // 2          # 50% overlap between frames
rms = librosa.feature.rms(y=filtered_audio, frame_length=frame_length, hop_length=hop_length)[0]
```

**What it does:**
Divides the filtered audio into short overlapping frames and computes the Root Mean Square energy of each frame. The result is a 1D array showing signal power over time — high values mean a tone is playing, low values mean silence.

**Why not use raw amplitude?**
Raw amplitude oscillates thousands of times per second (it is the audio wave). You cannot tell from a single sample whether a tone is on or off. RMS averages the energy over a short window, giving a smooth curve that clearly shows ON and OFF regions.

**Why 10ms frames?**
Morse dots at normal speed are roughly 60–120ms long. A 10ms frame is short enough to catch the beginning and end of a dot (6–12 frames per dot) but long enough to smooth out the wave oscillation. Shorter frames would be noisy; longer frames would blur short dots.

**Why 50% overlap?**
Overlapping frames give smoother time resolution. Without overlap, you get one measurement per 10ms. With 50% overlap, you get one measurement per 5ms, giving more accurate detection of where a tone starts and stops.

---

#### Step 2: Adaptive Threshold

```python
noise_floor = np.max(rms) * 0.05
active_rms  = rms[rms > noise_floor]
threshold   = np.median(active_rms) * 0.6
signal_on   = rms > threshold
```

**What it does:**
Converts the continuous RMS array into a binary array: `True` = tone is ON, `False` = tone is OFF.

**Why adaptive, not fixed?**
Different audio files have different overall loudness. A fixed threshold of 0.1 would work on a loud file but miss tones in a quiet file, and trigger false positives on a very loud file. An adaptive threshold scales with each specific recording.

**Why `noise_floor = max * 0.05`?**
Ignores the quietest 5% of the signal as background noise so it does not influence the threshold calculation.

**Why `median * 0.6`?**
The median of the active signal (samples louder than the noise floor) represents a typical "on" energy level. Setting the threshold at 60% of this catches the edges of tones as they fade in and out, without being so low that noise triggers false detections. The mean was not used because loud peaks would push it too high.

---

#### Step 3: Run-Length Segmentation

```python
segments = []  # list of (is_on: bool, duration: int, start_frame: int)
```

Scans the binary `signal_on[]` array and collapses consecutive identical values into segments. Each segment records whether it was ON or OFF, how long it lasted (in frames), and when it started.

**Why this step?**
K-Means needs a list of durations to cluster. You cannot run K-Means on the raw binary array — you need to first extract "how long was this tone on?" and "how long was this silence?". Run-length encoding does exactly that.

---

## 5. K-Means Clustering — Full Explanation

### What is K-Means?

K-Means is an unsupervised machine learning algorithm. You give it a list of numbers and tell it how many groups (k) to find. It finds cluster centers that minimize the distance between each data point and its nearest center.

It is "unsupervised" because you do not give it labeled training data — it discovers structure in the data itself.

### Why Use K-Means for Morse Decoding?

In Morse code:
- A **dot** is a short tone
- A **dash** is a long tone (theoretically 3× the length of a dot)

A naive approach would set a fixed threshold: "anything shorter than 150ms is a dot, longer is a dash." This fails on real audio because playback speed varies between recordings. One file might have 80ms dots; another might have 200ms dots.

K-Means learns the boundary from the actual durations in each specific file. It does not assume what "short" and "long" are — it discovers them.

---

### K-Means on ON Segments (Dot vs Dash)

```python
on_durations = np.array([d for is_on, d, _ in segments if is_on]).reshape(-1, 1)

km_on = KMeans(n_clusters=2, n_init=10, random_state=0)
km_on.fit(on_durations)

dot_cluster  = int(np.argmin(km_on.cluster_centers_))
dash_cluster = 1 - dot_cluster
```

**Why `n_clusters=2`?**
There are exactly two things a Morse ON segment can be: a dot or a dash. No more, no less.

**Why `n_init=10`?**
K-Means starts with random cluster centers. Different random starts can give different results (this is called a local minimum problem). Running 10 independent starts and picking the best result makes the clustering stable.

**Why `random_state=0`?**
Fixes the random seed so results are identical every time the same file is decoded. Without this, the dot/dash assignment could differ between runs.

**Why `dot_cluster = argmin(cluster_centers_)`?**
After fitting, K-Means does not label its clusters "dot" or "dash" — it just gives them numbers (0 and 1). But dots are always shorter, so the cluster whose center (average duration) is smaller must be the dot cluster. `argmin` finds the index of the smaller center automatically.

**How each segment is classified:**
```python
label = km_on.predict([[duration]])[0]
symbol = '.' if label == dot_cluster else '-'
```
Each ON segment's duration is passed to the trained model, which assigns it to the nearest cluster center.

---

### Ratio Check — Fallback When Clusters Are Too Close

```python
ratio = centers[dash_cluster] / (centers[dot_cluster] + 1e-9)
if ratio < 2.0:
    # fallback: split at mean instead of using K-Means
    _on_split = float(np.mean(on_durations))
    _use_kmeans_on = False
```

**Why this check?**
In real Morse code, dashes are 3× the length of dots. If K-Means finds two clusters whose centers are less than 2× apart, something went wrong — the audio may be too noisy, or there are only dots with no real dashes. In this case K-Means did not find a meaningful split, and using it would misclassify everything. The fallback (split at the mean) is simpler and safer.

---

### K-Means on OFF Segments (Gap Classification)

```python
n_off_clusters = min(3, len(off_durations))
km_off = KMeans(n_clusters=n_off_clusters, n_init=10, random_state=0)
km_off.fit(off_durations)
sorted_centers = np.sort(km_off.cluster_centers_.flatten())
```

Morse code has three types of silence:

| Silence Type | Meaning | Action |
|---|---|---|
| Short gap | Between dots/dashes within the same letter | Do nothing, keep building the letter |
| Medium gap | Between two letters | Decode current letter, start next |
| Long gap | Between two words | Decode current letter, add a space |

**Why `n_clusters=3`?**
There are exactly three gap types in Morse code.

**Why `min(3, len(off_durations))`?**
If the audio is very short and has fewer than 3 silence segments total, K-Means cannot form 3 clusters. Clamping to the actual count prevents a crash.

**Why `sorted_centers`?**
After K-Means, cluster indices are arbitrary (cluster 0 is not necessarily the shortest silence). Sorting the centers gives a reliable ordering:
- `sorted_centers[0]` = shortest (intra-letter gap)
- `sorted_centers[1]` = medium (letter gap)
- `sorted_centers[2]` = longest (word gap)

This ordering holds regardless of which cluster number K-Means assigned internally.

**How each silence is classified:**
```python
label  = km_off.predict([[duration]])[0]
center = km_off.cluster_centers_[label][0]

if center == sorted_centers[2]:    # word gap
    decode current letter, append space
elif center == sorted_centers[1]:  # letter gap
    decode current letter, start next
# else: intra-letter gap — do nothing
```

---

### Dictionary Lookup

```python
MORSE_CODE_DICT = {
    '.-': 'A', '-...': 'B', '-.-.': 'C', ...  # all 26 letters + 10 digits
}

letter = self.morse_code_dict.get(''.join(current_char), '?')
```

Dots and dashes accumulate in `current_char[]`. When a letter gap or word gap is detected, the accumulated pattern is looked up in the dictionary. Unknown patterns output `'?'`.

**Why a hardcoded dictionary?**
The Morse alphabet is a fixed international standard (ITU-R M.1677). It does not change. A dictionary lookup is O(1), simple, and perfectly readable.

**Why output `'?'` for unknown patterns?**
On noisy audio, K-Means may misclassify a segment, producing a dot/dash pattern that is not in the dictionary. Returning `'?'` makes errors visible rather than silently skipping letters, which helps diagnose decoding quality.

---

## 6. Libraries Used and Why

| Library | Purpose | Why This One |
|---|---|---|
| `librosa` | Audio loading, RMS feature extraction | Industry standard for audio ML in Python; handles WAV and MP3 in one call |
| `scipy` | Butterworth filter design (`butter`, `filtfilt`) | Gold standard signal processing library; `filtfilt` gives zero-phase filtering |
| `numpy` | Array operations throughout | Required by librosa and scipy; efficient numerical computation |
| `scikit-learn` | `KMeans` clustering | Standard ML library; clean API, reliable implementation |
| `matplotlib` | Waveform plot | Most common Python plotting library; integrates with tkinter via `FigureCanvasTkAgg` |
| `customtkinter` | Modern desktop GUI | Gives modern styled UI (rounded buttons, themes) unlike plain tkinter's dated appearance |
| `sounddevice` | Real-time audio playback | Low-latency output; works directly with NumPy arrays |
| `soundfile` | Reading audio data for playback | Reads WAV/MP3 into NumPy arrays cleanly |

---

## 7. Build and Distribution

**Why build a standalone `.exe`?**
Teammates and the professor may not have Python installed or may not know how to set up a virtual environment. A standalone executable means anyone can double-click and run the application immediately — no setup required.

**Why a custom `main.spec` for PyInstaller?**
The default `pyinstaller main.py` command misses hidden dependencies:
- `customtkinter` has asset folders (themes, fonts) that must be bundled manually
- `sounddevice` and `soundfile` require DLLs that are not auto-detected

The spec file explicitly declares all of these. It is version-controlled so any team member can rebuild the exe from source.

**Why `create_test.py`?**
A synthetic test file (SOS at 700 Hz, exact standard Morse timing) provides a known-correct input during development. If SOS does not decode correctly on clean synthetic audio, the bug is definitely in the code — not the input recording.

---

## 8. Known Limitation — Word Gap Detection

### What Goes Wrong

The K-Means classifier for OFF segments works by grouping silence durations into three clusters. On noisy or real-world audio, these clusters overlap — a word gap that should be 210ms might only last 130ms due to speaker speed, and K-Means places it in the "letter gap" cluster instead.

**Result:**

```
Intended:   HELLO WORLD
Decoded:    HELLOWORLD       <- word gap misclassified as letter gap
```

Or the reverse:

```
Intended:   HELLO
Decoded:    HE LLO           <- letter gap misclassified as word gap
```

### Why K-Means Struggles Here

K-Means needs clearly separated clusters. For ON segments (dots and dashes), the duration ratio is 1:3 — well separated. For OFF segments, real audio does not always preserve clean 1:3:7 gap ratios (intra-letter : letter : word). The clusters blur together and K-Means cannot reliably separate them.

---

## 9. Planned Improvement — MorseCorrector

### What It Is

A new class `MorseCorrector` that runs **after** the decoder has produced text. It takes each decoded word and checks whether it is a real English word. If not, it finds the closest real word using **edit distance** (Levenshtein distance).

### What Is Edit Distance?

Edit distance is the minimum number of single-character operations — insert, delete, or substitute — needed to transform one string into another.

```
HEL?O  ->  HELLO    edit distance = 1  (substitute ? with L)
WROLD  ->  WORLD    edit distance = 2  (rearrange letters)
HELO   ->  HELLO    edit distance = 1  (insert one L)
```

The corrector compares the decoder's output against every word in a dictionary and returns the word with the smallest edit distance.

### What It Fixes

Character-level errors within a word:
- Wrong letters (K-Means misclassified a dot as a dash)
- Missing letters (a segment was lost due to noise)
- `?` placeholders (dot/dash pattern not in the Morse dictionary)

**Before correction:** `H E L ? O   W R O L D`
**After correction:**  `H E L L O   W O R L D`

### What It Does NOT Fix

Word boundary errors. If `HELLO WORLD` decoded as `HELLOWORLD` (one word), the corrector receives one word and compares it to single dictionary words. It finds nothing close — because it does not know that `HELLOWORLD` should be split into two words. The boundary problem is invisible to edit distance.

---

## 10. Professor's Suggestion — AI Word Gap Prediction

### The Key Insight

The professor identified the actual root cause: the word boundary detection is wrong, not just the letters inside words. His suggestion is to stop relying on K-Means silence duration clustering for word boundaries, and instead use language awareness to determine where word gaps belong.

### Why This Is a Different Problem

Edit distance works on the **output** (fixing letters inside already-segmented words).
The professor's approach works on the **letter stream** (figuring out where words begin and end).

These are two different problems solved at two different stages.

---

### Approach A — Word Segmentation with Dynamic Programming

This is the most practical interpretation of the professor's suggestion.

You take the raw unsegmented letter output and run a word segmentation algorithm that splits it into real words.

**How it works:**

Given the string `HELLOWORLD`, try every possible split point and score it using a word frequency dictionary:

```
HELLOWORLD
|
Try: H + ELLOWORLD        H is a word (score: low), recurse on ELLOWORLD
Try: HE + LLOWORLD        HE is a word (score: medium), recurse on LLOWORLD
Try: HEL + LOWORLD        HEL is not a word, skip
Try: HELL + OWORLD        HELL is a word, recurse on OWORLD
Try: HELLO + WORLD        HELLO is a word (high score), WORLD is a word (high score)  <-- WINS

Output: HELLO WORLD
```

Each word is scored by how common it is in English. The split combination with the highest total score is chosen. This is dynamic programming — it avoids re-checking the same substring multiple times.

**This is considered AI** because it uses statistical knowledge of natural language (word frequencies) to make predictions about structure.

---

### Approach B — Language Model (More Advanced AI)

A more powerful version feeds the raw decoder output into a trained sequence model — an n-gram language model or a transformer — that predicts the most likely English word sequence.

```
Decoder output (letter stream):   H E L L O W O R L D
Language model asks:               What English sentence most likely produced these letters?
Model output:                      HELLO WORLD
```

This simultaneously fixes word boundaries AND character errors, because the model considers the whole sequence together.

**Tradeoff:** Requires a trained model, more complex to implement, may need internet access if using a large language model.

---

## 11. Comparison — Both Approaches

| | MorseCorrector (Edit Distance) | Professor's AI Word Gap |
|---|---|---|
| **Stage** | Post-decoding, per word | On raw letter stream, before word splitting |
| **What it fixes** | Wrong/missing characters inside a word | Wrong/missing word boundaries |
| **Addresses root cause?** | No — assumes boundaries are already right | Yes — fixes the segmentation itself |
| **Example input** | `HEL?O WROLD` | `HELLOWORLD` |
| **Example output** | `HELLO WORLD` | `HELLO WORLD` |
| **Works on** | Character errors | Boundary errors |
| **Technique** | Levenshtein distance against dictionary | Dynamic programming + word frequency list (or language model) |
| **Complexity** | Low | Medium to high |
| **Works offline?** | Yes | Yes (frequency list), No (online LLM) |
| **OOP fit** | New `MorseCorrector` class | New `WordSegmenter` class |

**Neither approach is a replacement for the other.** They fix different types of errors. The ideal system uses both.

---

## 12. Combined Ideal Architecture

The best solution separates the signal problem from the language problem:

```
Raw audio
    |
    v
Filter + RMS + K-Means
(dots and dashes only — stop relying on K-Means for word gaps)
    |
    v
Raw letter stream: H E L L O W O R L D S T O P
    |
    v
WordSegmenter (AI word gap prediction)
Dynamic programming over word frequency list
→ Insert word boundaries
→ HELLO WORLD STOP
    |
    v
MorseCorrector (Edit distance per word)
→ Fix remaining character errors
→ HELLO WORLD STOP
    |
    v
Final output displayed in UI
```

**Why this is better:**
K-Means is good at clustering signal durations (dot vs dash). It struggles at the language-level task of knowing where words end. By removing word gap detection from K-Means and delegating it to a language-aware model, each component does the job it is actually suited for.

---

## Project File Structure

```
morse_project/
├── main.py                  <- MorseApp class, entry point
├── src/
│   ├── __init__.py
│   ├── audio_loader.py      <- AudioLoader class
│   ├── signal_filter.py     <- SignalFilter class
│   ├── morse_decoder.py     <- MorseDecoder class (RMS + K-Means + dictionary)
│   ├── signal_visualizer.py <- SignalVisualizer class
│   └── ui_display.py        <- UIDisplay class (GUI + animation)
├── create_test.py           <- generates synthetic SOS test audio at 700 Hz
├── config.json              <- configurable parameters
├── pyproject.toml           <- project metadata and dependencies
├── main.spec                <- PyInstaller build configuration
├── dist/MorseDecoder/       <- compiled standalone .exe
├── docs/                    <- documentation folder
├── HOW_TO_RUN.md            <- setup guide for teammates
├── DEVLOG.md                <- developer log
└── PROJECT_REPORT.md        <- this file
```

---

## Git Development Timeline

| Commit | What Was Done |
|---|---|
| `3383885` | Initial Morse Code Decoder |
| `1f6f911` | Added developer log |
| `6bb83f0` | More developer log entries |
| `a62c940` | Improved decoder accuracy, added timed animation, upgraded UI |
| `7e374e5` | Added `pyproject.toml` with project metadata and dependencies |
| `3cdf8c9` | Restructured project into `src/` folder, added `config.json`, first exe build |
| `686f84b` | Fixed import paths in `ui_display.py` after restructuring |
| `e0c0630` | Rebuilt exe with correct PyInstaller spec (all dependencies bundled) |
| `3569270` | Added `HOW_TO_RUN.md` for teammates |
