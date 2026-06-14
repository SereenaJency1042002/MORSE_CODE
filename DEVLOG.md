# 📓 Developer Log — Morse Code Decoder
**Project:** Intelligent Morse Code Decoder  
**Course:** Object-Oriented Programming (OOP) — TH Köln  
**Team:** Sereena Jency 
**Semester:** Summer Semester 2026  

---

## Week 1 — May 24, 2026 — Project Setup & Architecture Design

### What We Did
- Read and analyzed the project assignment documents thoroughly
- Understood the core requirements: OOP design, real-time audio decoding, signal visualization, intelligent correction layer
- Designed the full class architecture — decided to split responsibilities into 6 separate classes
- Set up the Python development environment with virtual environment (venv)
- Installed required libraries: `librosa`, `customtkinter`, `numpy`, `scipy`, `matplotlib`, `scikit-learn`
- Created project folder structure with separate files for each class
- Created `.gitignore` to exclude `venv/` and `__pycache__/`

### Architecture Decision
We decided to follow strict OOP principles as required by the professor — one class per responsibility:
AudioLoader       → loads audio files
SignalFilter      → cleans and filters the signal
SignalVisualizer  → draws signal graph
MorseDecoder      → detects dots/dashes, converts to text
UIDisplay         → manages the UI window
MorseApp          → main controller class

### Challenges
- Understanding Morse code timing rules (dot = 1 unit, dash = 3 units, letter gap = 3 units, word gap = 7 units)
- Deciding how classes should communicate with each other

---

## Week 2 — May 25-26, 2026 — Core Implementation

### What We Did
- Implemented `AudioLoader` class using `librosa` to load WAV and MP3 files
- Implemented `SignalFilter` class using `scipy.signal.butter` and `filtfilt` — bandpass filter (600-1200 Hz) to isolate Morse tone
- Implemented `SignalVisualizer` class using `matplotlib` — plots time-domain signal with proper axis labels
- Implemented `MorseDecoder` class — initial version using basic threshold detection
- Implemented `UIDisplay` class using `customtkinter` — basic window with decoded text display
- Connected all classes in `main.py` via `MorseApp` controller class

### Technical Decisions
- Used bandpass filter at 600-1200 Hz because Morse CW signals typically transmit at 700 Hz
- Used `librosa` as the primary audio library as recommended by the professor in class
- Used `customtkinter` for UI as recommended by professor

### Testing
- Created `create_test.py` to generate clean synthetic SOS audio for testing
- First run: signal graph displayed correctly ✅
- Decoder output: initial issues with dot/dash detection

---

## Week 3 — May 27-29, 2026 — UI Improvement & Decoder Upgrade

### What We Did
- Completely redesigned `UIDisplay` to be a professional all-in-one window
- Embedded matplotlib signal graph directly inside the UI window (no separate popup)
- Added **Load Audio File** button with file dialog (supports WAV and MP3)
- Added **Decode** button that triggers the full pipeline
- Added proper window close handler to stop program correctly
- Upgraded `MorseDecoder` to use proper RMS envelope extraction via `librosa.magphase` and `librosa.feature.rms`
- Implemented K-Means clustering to automatically detect dot/dash durations
- Implemented K-Means clustering for OFF periods to detect symbol gaps, letter gaps and word gaps
- Added dark-themed signal graph (cyan signal on dark background)

### Technical Decisions
- Switched from raw STFT energy to `librosa.magphase` + `librosa.feature.rms` for more accurate energy detection
- Used `0.5 * max(envelope)` as threshold — proven method from research
- Used K-Means with 2 clusters for ON periods (dot vs dash) and up to 3 clusters for OFF periods
- Moved all decoding logic inside `UIDisplay.decode()` method so UI controls the full pipeline

### Results
- ✅ SOS decoded correctly from clean synthetic audio
- ✅ Professional UI with embedded graph working
- ✅ File loader working — can load any WAV or MP3 file
- ⚠️ Real audio from morsecode.world partially working (some letters incorrect)

### Challenges
- K-Means clustering boundary between dots and dashes varies depending on audio quality
- Letter gap detection inconsistent for real audio files
- Real audio timing is less precise than synthetic audio

---

## Week 4 — May 29, 2026 — Git Setup & Research

### What We Did
- Pushed full codebase to personal GitHub repository
- Set up `.gitignore` properly (excludes `venv/`, `__pycache__/`, `audio_files/`)
- Researched existing working Morse audio decoder projects for reference:
  - `github.com/mkouhia/morse-audio-decoder`
  - `github.com/ggerganov/ggmorse`
  - `github.com/joseph-crowley/morse-audio`

### Key Research Findings
- Proper approach uses **moving RMS with Hann window** (10ms windows)
- **Median-based adaptive threshold** is more robust for noisy audio
- Intelligent correction layer using N-grams or HMM is essential for real-world signals

---

---

## Week 5 — June 14, 2026 — Decoder Upgrade & Real Audio Improvements

### What We Did

#### 1. Finer RMS Envelope (10ms Hann Window)
Replaced STFT-based RMS with a direct windowed RMS using a 10ms frame:
```python
frame_length = max(64, int(self.sr * 0.01))  # 10ms
hop_length = frame_length // 2
rms = librosa.feature.rms(y=self.filtered_audio, frame_length=frame_length, hop_length=hop_length)[0]
```
**Why:** Default STFT gives ~23ms per frame at 22050 Hz — too coarse for fast Morse. 10ms gives ~110 data points per second, so short dots are no longer missed.

#### 2. Adaptive Median Threshold
Replaced the fixed `0.5 * max(rms)` threshold with a median-based approach:
```python
noise_floor = np.max(rms) * 0.05
active_rms = rms[rms > noise_floor]
threshold = np.median(active_rms) * 0.6 if len(active_rms) > 0 else np.max(rms) * 0.5
```
**Why:** A single noise spike could raise `max(rms)` and push the threshold too high, causing the decoder to see silence where there was signal. The median of active (non-silent) frames is much more stable.

#### 3. Decode with Timing (`decode_with_timing()`)
Added a new method that returns timed events alongside the decoded text:
```python
# Each segment now tracks its start frame index
segments.append((bool(current), count, start))

# Time in seconds for each event
time_sec = (start_frame * hop_length) / self.sr
events.append((time_sec, 'symbol', symbol))   # dot or dash
events.append((time_sec, 'letter', letter))   # decoded letter
events.append((time_sec, 'word', ' '))        # word space
```
Returns `(events, full_text)`. The `decode()` method now calls this internally:
```python
def decode(self):
    _, text = self.decode_with_timing()
    return text
```

#### 4. K-Means Ratio Validation (L detection fix)
After clustering ON durations, validate that the dot/dash clusters are in the expected ~3:1 ratio. If not, fall back to a mean-based split:
```python
ratio = centers[dash_cluster] / (centers[dot_cluster] + 1e-9)
if ratio < 2.0:
    _on_split = float(np.mean(on_durations))
    _use_kmeans_on = False
else:
    _use_kmeans_on = True
```
**Why:** In dot-heavy audio (many E, I, S, H), K-Means clusters drift and the boundary between dots and dashes becomes unclear. This caused `L` (`.-..`) to be decoded as `?` because the dash was misclassified as a dot. The mean-based fallback splits durations at their natural midpoint.

#### 5. Background Thread Processing
Moved all heavy computation (librosa load, bandpass filter, KMeans decode) into a background thread so the UI stays responsive:
```python
def _process():
    y, sr = AudioLoader(self.audio_file).load()
    filtered = SignalFilter(y, sr).filter()
    events, _ = MorseDecoder(filtered, sr, MORSE_CODE_DICT).decode_with_timing()
    self.app.after(0, lambda: self._on_decode_ready(session, y, sr, filtered, events))

threading.Thread(target=_process, daemon=True).start()
```
**Important:** `plt.subplots()` must stay on the main thread — only pure computation goes in the background thread.

### Results
- ✅ L (`.-..`) now correctly detected in real audio
- ✅ Adaptive threshold handles noisy/low-volume files
- ✅ Finer RMS captures fast Morse signals accurately
- ✅ UI no longer freezes during processing
- ⚠️ Word gap detection still inconsistent on some real audio files

---

## 📋 Current Status

| Feature | Status |
|---------|--------|
| Audio loading (WAV/MP3) | ✅ Done |
| Bandpass signal filtering | ✅ Done |
| Signal visualization (time-domain) | ✅ Done |
| Morse decoding — clean audio | ✅ Done |
| Professional UI with file loader | ✅ Done |
| Finer RMS envelope (10ms window) | ✅ Done |
| Adaptive median threshold | ✅ Done |
| Decode with timing (timed events) | ✅ Done |
| K-Means ratio validation fallback | ✅ Done |
| Audio playback in UI | ✅ Done |
| Background thread processing | ✅ Done |
| Morse decoding — real/noisy audio | ⚠️ Partial |
| Variable speed detection | ⬜ Planned |
| Intelligent correction layer | ⬜ Planned |

---

## 🔜 Next Steps

1. Fix word gap detection for real-world audio
2. Build intelligent correction layer using N-grams
3. Add variable speed (WPM) detection
4. Push to RWTH GitLab once access is granted
5. Prepare Presentation 1 slides (June 29, 2026)

---

## 📚 References

- [librosa Documentation](https://librosa.org/doc/latest/)
- [morse-audio-decoder by mkouhia](https://github.com/mkouhia/morse-audio-decoder)
- [ggmorse real-time decoder](https://github.com/ggerganov/ggmorse)
- [Morse Code Ninja](https://morsecode.ninja/resources/index.html)
- [scipy.signal Documentation](https://docs.scipy.org/doc/scipy/reference/signal.html)
- [customtkinter Documentation](https://customtkinter.tomschimansky.com/)