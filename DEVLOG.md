# 📓 Developer Log — Morse Code Decoder
**Project:** Intelligent Morse Code Decoder  
**Course:** Object-Oriented Programming (OOP) — TH Köln  
**Team:** Sereena Jency + Partner  
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

## 📋 Current Status

| Feature | Status |
|---------|--------|
| Audio loading (WAV/MP3) | ✅ Done |
| Bandpass signal filtering | ✅ Done |
| Signal visualization (time-domain) | ✅ Done |
| Morse decoding — clean audio | ✅ Done |
| Professional UI with file loader | ✅ Done |
| Morse decoding — real/noisy audio | ⚠️ Partial |
| Intelligent correction layer | ⬜ Planned |
| Variable speed detection | ⬜ Planned |
| Audio playback in UI | ⬜ Planned |

---

## 🔜 Next Steps

1. Implement proper Hann window RMS envelope extraction
2. Build intelligent correction layer using N-grams
3. Add variable speed (WPM) detection
4. Add audio playback feature in UI
5. Push to RWTH GitLab once access is granted
6. Prepare Presentation 1 slides (June 29, 2026)

---

## 📚 References

- [librosa Documentation](https://librosa.org/doc/latest/)
- [morse-audio-decoder by mkouhia](https://github.com/mkouhia/morse-audio-decoder)
- [ggmorse real-time decoder](https://github.com/ggerganov/ggmorse)
- [Morse Code Ninja](https://morsecode.ninja/resources/index.html)
- [scipy.signal Documentation](https://docs.scipy.org/doc/scipy/reference/signal.html)
- [customtkinter Documentation](https://customtkinter.tomschimansky.com/)