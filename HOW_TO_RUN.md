# How to Run — Intelligent Morse Code Decoder

## Option A: Run the EXE (easiest, no Python needed)

1. Navigate to `dist\MorseDecoder\`
2. Double-click `MorseDecoder.exe`
3. The app window opens — done.

> The EXE is self-contained. No installation required.

---

## Option B: Run from Source (Python)

### Prerequisites

- Python 3.11 or higher — download from https://www.python.org/downloads/
- Git — download from https://git-scm.com/downloads

### Step 1 — Clone the repository

```
git clone https://github.com/SereenaJency1042002/MORSE_CODE.git
```

```
cd MORSE_CODE
```

### Step 2 — Create a virtual environment

```
python -m venv venv
```

### Step 3 — Activate the virtual environment

**Windows (PowerShell):**
```
venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```
venv\Scripts\activate.bat
```

> You should see `(venv)` appear at the start of your terminal line after activation.

### Step 4 — Install dependencies

```
pip install .
```

### Step 5 — Run the app

```
python main.py
```

The GUI window will open.

---

## Using the App

1. Click **Load Audio File** and select a `.wav` or `.mp3` file
2. Click **Decode** — the signal graph appears and decoding runs
3. The decoded text is shown in the output panel

To generate a test SOS audio file to try it out:

```
python create_test.py
```

This creates a file in `audio_files/` — load it into the app to test.

---

## Project Structure

```
MORSE_CODE/
  main.py              # entry point — launches the app
  create_test.py       # generates a test SOS WAV file
  pyproject.toml       # project metadata and dependencies
  config.json          # app configuration
  src/
    audio_loader.py    # loads WAV/MP3 files
    signal_filter.py   # bandpass filter (600-1200 Hz)
    signal_visualizer.py  # matplotlib graph in the UI
    morse_decoder.py   # RMS + K-Means → dots/dashes → text
    ui_display.py      # customtkinter GUI
  audio_files/         # put your test audio files here
  dist/MorseDecoder/   # compiled EXE lives here
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Make sure venv is activated and `pip install .` was run |
| PowerShell blocks Activate.ps1 | Run: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| App won't open on Windows | Try right-click → Run as Administrator |
| EXE flagged by antivirus | This is a false positive common with PyInstaller — allow it |
| Audio file not loading | Only `.wav` and `.mp3` are supported |
| Decoding gives wrong output | Try with the generated test file first (`python create_test.py`) |
