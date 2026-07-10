import random
import threading
import time
from tkinter import filedialog

import customtkinter as ctk
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import LinearSegmentedColormap

ctk.set_appearance_mode("dark")

# Dark radio-style palette
_BG        = "#0a0a0a"   # main background
_PANEL     = "#111111"   # left panel background
_BORDER    = "#222222"   # hairline borders
_BORDER2   = "#333333"   # button outline (neutral)
_GREEN     = "#00ff41"   # primary accent
_GREEN_LINE = "#00c835"  # waveform line
_ORANGE    = "#ff6600"   # secondary accent
_RED       = "#ff4444"   # stop / alert
_CYAN      = "#00ccff"   # AI predicted text
_MUTED     = "#555555"   # small caption labels
_MUTED2    = "#444444"   # section labels / axes
_TEXT      = "#cccccc"   # neutral button text

_SDR_CMAP = LinearSegmentedColormap.from_list(
    "sdr", ["#000000", "#001a33", "#ff6600", "#ffff00"]
)


class UIDisplay:
    def __init__(self, groq_api_key=None):
        self.groq_api_key = groq_api_key
        self.audio_file = None
        self._playback_thread = None
        self._stop_playback = False
        self._decode_session = 0
        self._current_morse = ""
        self._playhead = None
        self._canvas_ref = None
        self._decode_start_time = 0
        self._audio_duration = 0
        self.ai_visible = False
        self._loading = False
        self._smeter_active = False
        self._pulse_on = False

        self.app = ctk.CTk()
        self.app.title("CW Decoder — Morse Code Decoder")
        self.app.geometry("1200x700")
        self.app.minsize(1200, 700)
        self.app.configure(fg_color=_BG)

        self._build_left_panel()
        self._build_right_panel()
        self._animate_smeter()
        self._animate_status_pulse()

        self.app.protocol("WM_DELETE_WINDOW", self.on_close)

    # ── small helpers ─────────────────────────────────────────────────────────

    def _spaced(self, text):
        return " ".join(list(text))

    def _make_outline_button(self, parent, text, command, border_color, text_color=None, state="normal"):
        text_color = text_color or border_color
        btn = ctk.CTkButton(
            parent, text=text, command=command, state=state,
            fg_color=_BG, hover_color="#151515",
            border_width=1, border_color=border_color, text_color=text_color,
            font=ctk.CTkFont(family="Courier New", size=11, weight="bold"),
            corner_radius=4, height=32,
        )

        def _on_enter(_e):
            btn.configure(border_color=_GREEN, text_color=_GREEN)

        def _on_leave(_e):
            btn.configure(border_color=border_color, text_color=text_color)

        btn.bind("<Enter>", _on_enter)
        btn.bind("<Leave>", _on_leave)
        return btn

    def _set_status(self, text, color):
        self.status_label.configure(text=text, text_color=color)
        self.status_dot.configure(text_color=color)

    # ── left panel ────────────────────────────────────────────────────────────

    def _build_left_panel(self):
        self.left_panel = ctk.CTkFrame(self.app, fg_color=_PANEL, width=360, corner_radius=0)
        self.left_panel.pack(side="left", fill="y")
        self.left_panel.pack_propagate(False)

        # Title
        title_frame = ctk.CTkFrame(self.left_panel, fg_color=_PANEL)
        title_frame.pack(fill="x")
        ctk.CTkLabel(
            title_frame, text="C W  D E C O D E R",
            font=ctk.CTkFont(size=22, weight="bold"), text_color=_GREEN,
        ).pack(pady=(20, 10))
        ctk.CTkFrame(title_frame, fg_color=_BORDER, height=1).pack(fill="x")

        # Frequency display
        freq_box = ctk.CTkFrame(self.left_panel, fg_color=_BG, border_color=_BORDER, border_width=1, corner_radius=6)
        freq_box.pack(fill="x", padx=16, pady=(16, 10))
        ctk.CTkLabel(freq_box, text=self._spaced("FREQUENCY"), font=ctk.CTkFont(size=9), text_color=_MUTED).pack(pady=(8, 0))
        self.freq_entry = ctk.CTkEntry(
            freq_box, fg_color=_BG, border_width=0, justify="center",
            font=ctk.CTkFont(family="Courier New", size=32, weight="bold"),
            text_color=_ORANGE,
        )
        self.freq_entry.insert(0, "14.020")
        self.freq_entry.pack()
        ctk.CTkLabel(freq_box, text="MHz", font=ctk.CTkFont(size=9), text_color=_MUTED).pack(pady=(0, 8))

        # WPM / SIGNAL
        metrics_frame = ctk.CTkFrame(self.left_panel, fg_color=_PANEL)
        metrics_frame.pack(fill="x", padx=16, pady=(0, 10))
        metrics_frame.columnconfigure(0, weight=1)
        metrics_frame.columnconfigure(1, weight=1)

        wpm_box = ctk.CTkFrame(metrics_frame, fg_color=_BG, border_color=_BORDER, border_width=1, corner_radius=6)
        wpm_box.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        ctk.CTkLabel(wpm_box, text=self._spaced("WPM"), font=ctk.CTkFont(size=9), text_color=_MUTED).pack(pady=(6, 0))
        self.wpm_value_label = ctk.CTkLabel(wpm_box, text="---", font=ctk.CTkFont(size=20, weight="bold"), text_color=_GREEN)
        self.wpm_value_label.pack(pady=(0, 6))

        signal_box = ctk.CTkFrame(metrics_frame, fg_color=_BG, border_color=_BORDER, border_width=1, corner_radius=6)
        signal_box.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        ctk.CTkLabel(signal_box, text=self._spaced("SIGNAL"), font=ctk.CTkFont(size=9), text_color=_MUTED).pack(pady=(6, 0))
        self.signal_value_label = ctk.CTkLabel(signal_box, text="S--", font=ctk.CTkFont(size=20, weight="bold"), text_color=_ORANGE)
        self.signal_value_label.pack(pady=(0, 6))

        # S-meter
        smeter_frame = ctk.CTkFrame(self.left_panel, fg_color=_PANEL)
        smeter_frame.pack(fill="x", padx=16, pady=(0, 10))
        ctk.CTkLabel(smeter_frame, text=self._spaced("S-METER"), font=ctk.CTkFont(size=9), text_color=_MUTED).pack(anchor="w")

        bars_row = ctk.CTkFrame(smeter_frame, fg_color=_PANEL)
        bars_row.pack(fill="x", pady=(4, 0))
        heights = [6, 8, 10, 12, 14, 17, 20, 23, 26]
        self._smeter_bright = [_GREEN] * 5 + ["#ffaa00"] * 2 + [_RED] * 2
        self._smeter_dim = ["#0a2e14"] * 5 + ["#332200"] * 2 + ["#330a0a"] * 2
        self.smeter_bars = []
        for h, dim_c in zip(heights, self._smeter_dim):
            bar = ctk.CTkFrame(bars_row, fg_color=dim_c, width=12, height=h, corner_radius=1)
            bar.pack(side="left", anchor="s", padx=1)
            bar.pack_propagate(False)
            self.smeter_bars.append(bar)

        ctk.CTkLabel(
            smeter_frame, text="1     3     5     7     9",
            font=ctk.CTkFont(size=8, family="Courier New"), text_color=_MUTED,
        ).pack(anchor="w", pady=(2, 0))

        # Device selector
        device_frame = ctk.CTkFrame(self.left_panel, fg_color=_PANEL)
        device_frame.pack(fill="x", padx=16, pady=(0, 10))
        ctk.CTkLabel(device_frame, text=self._spaced("INPUT DEVICE"), font=ctk.CTkFont(size=9), text_color=_MUTED).pack(anchor="w")
        self.device_menu = ctk.CTkOptionMenu(
            device_frame, values=["Loading..."],
            fg_color=_BG, button_color=_BG, button_hover_color="#151515",
            text_color=_GREEN, font=ctk.CTkFont(family="Courier New", size=10),
            dropdown_fg_color=_BG, dropdown_text_color=_GREEN,
        )
        self.device_menu.pack(fill="x", pady=(4, 0))
        self._populate_devices()

        # Buttons
        btn_stack = ctk.CTkFrame(self.left_panel, fg_color=_PANEL)
        btn_stack.pack(fill="x", padx=16, pady=(4, 10))

        self.load_btn = self._make_outline_button(btn_stack, "LOAD FILE", self.load_file, _BORDER2, _TEXT)
        self.load_btn.pack(fill="x", pady=3)

        self.decode_btn = self._make_outline_button(btn_stack, "DECODE & PLAY", self.decode, _BORDER2, _TEXT)
        self.decode_btn.pack(fill="x", pady=3)

        self.play_btn = self._make_outline_button(btn_stack, "PLAY AUDIO", self.play_audio, _BORDER2, _TEXT, state="disabled")
        self.play_btn.pack(fill="x", pady=3)

        self.start_live_btn = self._make_outline_button(btn_stack, "START LIVE", self.start_live, _GREEN, _GREEN)
        self.start_live_btn.pack(fill="x", pady=3)

        self.stop_btn = self._make_outline_button(btn_stack, "STOP", self.stop_audio, _RED, _RED, state="disabled")
        self.stop_btn.pack(fill="x", pady=3)

        # File status
        self.file_label = ctk.CTkLabel(
            self.left_panel, text="No file loaded",
            font=ctk.CTkFont(size=10, family="Courier New"), text_color=_MUTED,
            wraplength=320, justify="left",
        )
        self.file_label.pack(fill="x", padx=16, pady=(0, 10))

        # AI toggle
        self.ai_btn = self._make_outline_button(self.left_panel, "◈ SHOW AI PREDICTION", self.toggle_ai, _ORANGE, _ORANGE)
        self.ai_btn.pack(fill="x", padx=16, pady=(0, 10))

        # Status bar (pinned to bottom)
        status_bar = ctk.CTkFrame(self.left_panel, fg_color=_PANEL)
        status_bar.pack(side="bottom", fill="x", padx=16, pady=(8, 16))
        self.status_dot = ctk.CTkLabel(status_bar, text="●", font=ctk.CTkFont(size=10), text_color=_GREEN, width=14)
        self.status_dot.pack(side="left")
        self.status_label = ctk.CTkLabel(status_bar, text="READY", font=ctk.CTkFont(size=10), text_color=_MUTED)
        self.status_label.pack(side="left")

    def _populate_devices(self):
        try:
            devices = sd.query_devices()
            input_devices = [(i, d["name"]) for i, d in enumerate(devices) if d["max_input_channels"] > 0]
            if not input_devices:
                raise RuntimeError("No input devices found")
            names = []
            for idx, name in input_devices:
                label = f"★ {name}" if "cable" in name.lower() else name
                names.append(label)
            self.device_menu.configure(values=names)
            self.device_menu.set(names[0])
        except Exception:
            self.device_menu.configure(values=["FILE MODE ONLY"], state="disabled")
            self.device_menu.set("FILE MODE ONLY")

    def start_live(self):
        self._set_status("LIVE MODE — connect VB-Cable", _ORANGE)

    # ── right panel ───────────────────────────────────────────────────────────

    def _build_right_panel(self):
        self.right_panel = ctk.CTkFrame(self.app, fg_color=_BG, corner_radius=0)
        self.right_panel.pack(side="left", fill="both", expand=True)
        self.right_panel.columnconfigure(0, weight=1)
        self.right_panel.rowconfigure(0, weight=35)
        self.right_panel.rowconfigure(1, weight=38)
        self.right_panel.rowconfigure(2, weight=27)

        self._build_waveform()
        self._build_waterfall()
        self._build_text_boxes()

    def _build_waveform(self):
        wave_frame = ctk.CTkFrame(self.right_panel, fg_color=_BG)
        wave_frame.grid(row=0, column=0, sticky="nsew", padx=12, pady=(12, 4))
        ctk.CTkLabel(wave_frame, text=self._spaced("SIGNAL WAVEFORM"), font=ctk.CTkFont(size=9), text_color=_MUTED2).pack(anchor="w")

        self.graph_frame = ctk.CTkFrame(wave_frame, fg_color=_BG, border_color=_BORDER, border_width=1, corner_radius=6)
        self.graph_frame.pack(fill="both", expand=True, pady=(2, 0))

    def _build_waterfall(self):
        water_frame = ctk.CTkFrame(self.right_panel, fg_color=_BG)
        water_frame.grid(row=1, column=0, sticky="nsew", padx=12, pady=4)
        ctk.CTkLabel(water_frame, text=self._spaced("WATERFALL"), font=ctk.CTkFont(size=9), text_color=_MUTED2).pack(anchor="w")

        self.waterfall_container = ctk.CTkFrame(water_frame, fg_color=_BG, border_color=_BORDER, border_width=1, corner_radius=6)
        self.waterfall_container.pack(fill="both", expand=True, pady=(2, 0))
        self._build_waterfall_idle()

    def _build_waterfall_idle(self):
        fig, ax = plt.subplots(figsize=(8, 2.6))
        fig.patch.set_facecolor(_BG)
        ax.set_facecolor(_BG)
        ax.imshow(np.zeros((10, 10)), cmap=_SDR_CMAP, aspect="auto")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(_BORDER)
        fig.tight_layout(pad=0.5)

        canvas = FigureCanvasTkAgg(fig, master=self.waterfall_container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        self.waterfall_fig = fig
        self.waterfall_ax = ax
        self.waterfall_canvas = canvas

    def _update_waterfall(self, audio, sr):
        try:
            n_fft, hop = 512, 256
            if len(audio) <= n_fft:
                return
            frames = np.array([audio[i:i + n_fft] for i in range(0, len(audio) - n_fft, hop)])
            stft = np.abs(np.fft.rfft(frames, axis=1))
            spec_db = 20 * np.log10(stft + 1e-9)
            freqs = np.fft.rfftfreq(n_fft, 1 / sr)

            # Clip to a realistic dynamic range near the peak — without this,
            # the near-silent noise floor (down at the -180dB epsilon) stretches
            # the auto color scale so much that ordinary background energy already
            # falls in the colormap's orange/yellow range, making the tone look
            # far wider than it actually is.
            vmax = float(spec_db.max())
            vmin = vmax - 50

            self.waterfall_ax.clear()
            self.waterfall_ax.imshow(
                spec_db, origin="upper", aspect="auto", cmap=_SDR_CMAP,
                extent=[freqs[0], freqs[-1], spec_db.shape[0], 0],
                vmin=vmin, vmax=vmax,
            )

            # The signal is already bandpass-filtered to ~150Hz either side of the
            # tone before it gets here, so most of the 0..sr/2 axis is empty air.
            # Zoom to the active band instead of always showing the full range.
            peak_freq = freqs[np.argmax(spec_db.max(axis=0))]
            zoom_half_width = 400
            low = max(freqs[0], peak_freq - zoom_half_width)
            high = min(freqs[-1], peak_freq + zoom_half_width)
            self.waterfall_ax.set_xlim(low, high)

            self.waterfall_ax.set_xlabel("Frequency (Hz)", color=_MUTED2, fontsize=7)
            self.waterfall_ax.set_yticks([])
            self.waterfall_ax.tick_params(colors="#333333", labelsize=7)
            for spine in self.waterfall_ax.spines.values():
                spine.set_color(_BORDER)
            self.waterfall_fig.patch.set_facecolor(_BG)
            self.waterfall_ax.set_facecolor(_BG)
            self.waterfall_fig.tight_layout(pad=0.5)
            self.waterfall_canvas.draw_idle()
        except Exception as exc:
            print(f"Waterfall update error: {exc}")

    def _build_text_boxes(self):
        text_frame = ctk.CTkFrame(self.right_panel, fg_color=_BG)
        text_frame.grid(row=2, column=0, sticky="nsew", padx=12, pady=(4, 12))
        text_frame.columnconfigure(0, weight=1)
        text_frame.columnconfigure(1, minsize=1)
        text_frame.columnconfigure(2, weight=1)
        text_frame.rowconfigure(0, weight=1)

        raw_col = ctk.CTkFrame(text_frame, fg_color=_BG)
        raw_col.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        ctk.CTkLabel(raw_col, text=self._spaced("RAW DECODE"), font=ctk.CTkFont(size=9), text_color=_MUTED2).pack(anchor="w")
        self.morse_label = ctk.CTkLabel(
            raw_col, text="", font=ctk.CTkFont(size=18, family="Courier New", weight="bold"), text_color=_ORANGE,
        )
        self.morse_label.pack(anchor="w")
        self.text_box = ctk.CTkTextbox(
            raw_col, fg_color=_BG, text_color=_GREEN, border_width=0,
            font=ctk.CTkFont(family="Courier New", size=12),
        )
        self.text_box.pack(fill="both", expand=True, pady=(2, 0))

        ctk.CTkFrame(text_frame, fg_color=_BORDER, width=1).grid(row=0, column=1, sticky="ns")

        self.ai_frame = ctk.CTkFrame(text_frame, fg_color=_BG)
        self.ai_frame.grid(row=0, column=2, sticky="nsew", padx=(8, 0))
        ctk.CTkLabel(self.ai_frame, text=self._spaced("AI PREDICTED"), font=ctk.CTkFont(size=9), text_color=_MUTED2).pack(anchor="w")
        self.ai_text_box = ctk.CTkTextbox(
            self.ai_frame, fg_color=_BG, text_color=_CYAN, border_width=0,
            font=ctk.CTkFont(family="Courier New", size=12),
        )
        self.ai_text_box.pack(fill="both", expand=True, pady=(2, 0))
        self.ai_frame.grid_remove()

    # ── animations ────────────────────────────────────────────────────────────

    def _animate_smeter(self):
        lit = random.randint(4, 7) if self._smeter_active else 0
        for i, bar in enumerate(self.smeter_bars):
            bar.configure(fg_color=self._smeter_bright[i] if i < lit else self._smeter_dim[i])
        self.signal_value_label.configure(text=f"S{lit}" if lit else "S--")
        self.app.after(200, self._animate_smeter)

    def _animate_status_pulse(self):
        self._pulse_on = not self._pulse_on
        base_color = self.status_label.cget("text_color")
        self.status_dot.configure(text_color=base_color if self._pulse_on else _MUTED2)
        self.app.after(600, self._animate_status_pulse)

    # ── WPM ───────────────────────────────────────────────────────────────────

    def _estimate_wpm(self, events):
        symbol_times = [t for t, et, _ in events if et == "symbol"]
        if len(symbol_times) < 2:
            return None
        deltas = [b - a for a, b in zip(symbol_times, symbol_times[1:]) if b - a > 0]
        if not deltas:
            return None
        unit = min(deltas)
        if unit <= 0:
            return None
        wpm = round(1.2 / unit)
        return max(5, min(60, wpm))

    # ── file loading ──────────────────────────────────────────────────────────

    def load_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Audio Files", "*.wav *.mp3 *.oga *.ogg")]
        )
        if file_path:
            self.audio_file = file_path
            name = file_path.replace("\\", "/").split("/")[-1]
            self.file_label.configure(text=f"● {name}", text_color=_GREEN)
            self.play_btn.configure(state="normal")

    # ── playback helpers ──────────────────────────────────────────────────────

    def play_audio(self):
        if not self.audio_file:
            return
        self._cancel_animation()
        sd.stop()
        self._stop_playback = False
        self.play_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self._smeter_active = True
        self._set_status("PLAYING", _GREEN)

        def _play():
            try:
                data, samplerate = sf.read(self.audio_file)
                sd.play(data, samplerate)
                sd.wait()
            finally:
                self.app.after(0, self._on_playback_done)

        self._playback_thread = threading.Thread(target=_play, daemon=True)
        self._playback_thread.start()

    def stop_audio(self):
        self._cancel_animation()
        sd.stop()
        self._stop_playback = True
        self._on_playback_done()

    def _on_playback_done(self):
        self.play_btn.configure(state="normal" if self.audio_file else "disabled")
        self.stop_btn.configure(state="disabled")
        self._smeter_active = False
        self._set_status("READY", _MUTED)

    def _cancel_animation(self):
        self._loading = False
        self._decode_session += 1
        self._current_morse = ""
        self.morse_label.configure(text="")

    def _start_spinner(self, session):
        self._loading = True
        frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        idx = [0]

        def _tick():
            if not self._loading or self._decode_session != session:
                return
            self.morse_label.configure(
                text=f"{frames[idx[0] % len(frames)]}   Processing..."
            )
            idx[0] += 1
            self.app.after(80, _tick)

        _tick()

    def _stop_spinner(self):
        self._loading = False
        self.morse_label.configure(text="")

    # ── moving playhead ───────────────────────────────────────────────────────

    def _update_playhead(self, session):
        if self._decode_session != session or self._playhead is None:
            return
        elapsed = time.time() - self._decode_start_time
        x = min(elapsed, self._audio_duration)
        self._playhead.set_xdata([x, x])
        self._canvas_ref.draw_idle()
        if elapsed < self._audio_duration:
            self.app.after(100, lambda s=session: self._update_playhead(s))

    # ── animated decode callbacks ─────────────────────────────────────────────

    def _show_symbol(self, session, symbol):
        if self._decode_session != session:
            return
        self._current_morse += symbol
        self.morse_label.configure(text=self._current_morse)

    def _show_letter(self, session, letter):
        if self._decode_session != session:
            return
        self._current_morse = ""
        self.morse_label.configure(text="")
        self.text_box.insert("end", letter)

    def _show_word(self, session):
        if self._decode_session != session:
            return
        self._current_morse = ""
        self.morse_label.configure(text="")
        self.text_box.insert("end", " ")

    # ── main decode ───────────────────────────────────────────────────────────

    def decode(self):
        if not self.audio_file:
            self.file_label.configure(text="⚠ Please load a file first!", text_color=_RED)
            return

        self._cancel_animation()
        sd.stop()
        session = self._decode_session

        # Show spinner immediately while processing runs in background
        self.decode_btn.configure(state="disabled", text="PROCESSING...")
        self.text_box.delete("1.0", "end")
        self.ai_text_box.delete("1.0", "end")
        self._start_spinner(session)
        self._smeter_active = True
        self._set_status("PROCESSING...", _ORANGE)

        def _process():
            from src.audio_loader import AudioLoader
            from src.signal_filter import SignalFilter
            from src.morse_decoder import MorseDecoder, MORSE_CODE_DICT
            from src.intelligent_corrector import IntelligentCorrector
            from src.ai_predictor import AIPredictor

            y, sr = AudioLoader(self.audio_file).load()
            filtered = SignalFilter(y, sr).filter()
            events, raw_text = MorseDecoder(filtered, sr, MORSE_CODE_DICT).decode_with_timing()

            corrected_events, corrected_text = IntelligentCorrector().correct(events, raw_text)

            raw_morse = ' '.join(d for _, et, d in corrected_events if et == 'symbol')

            if self.groq_api_key:
                final_text = AIPredictor(self.groq_api_key).correct(corrected_text, raw_morse)
            else:
                final_text = corrected_text

            # Pre-read original audio here (background thread) so _play() has no I/O delay
            raw_data, raw_sr = sf.read(self.audio_file)

            # Pass filtered audio back to main thread for figure creation
            self.app.after(0, lambda: self._on_decode_ready(session, y, sr, filtered, corrected_events, final_text, raw_data, raw_sr))

        threading.Thread(target=_process, daemon=True).start()

    def _set_final_text(self, session, text):
        if self._decode_session != session:
            return
        self.ai_text_box.delete("1.0", "end")
        self.ai_text_box.insert("end", text)

    def _on_decode_ready(self, session, y, sr, filtered, events, final_text="", raw_data=None, raw_sr=None):
        if self._decode_session != session:
            return

        self._stop_spinner()
        self.decode_btn.configure(state="normal", text="DECODE & PLAY")

        # Create figure on main thread (matplotlib requires this)
        from src.signal_visualizer import SignalVisualizer
        fig = SignalVisualizer(filtered, sr).get_figure()

        # Embed graph
        ax = fig.axes[0]
        ax.set_title('')
        ax.lines[0].set_color(_GREEN_LINE)
        ax.lines[0].set_linewidth(1)
        fig.patch.set_facecolor(_BG)
        ax.set_facecolor(_BG)
        ax.xaxis.label.set_color(_MUTED2)
        ax.yaxis.label.set_color(_MUTED2)
        ax.tick_params(colors="#333333", labelsize=8)
        ax.grid(True, color=(0, 1, 0.25, 0.07))
        for spine in ax.spines.values():
            spine.set_color(_BORDER)
        self._playhead = ax.axvline(x=0, color="#ffffff", linewidth=1.2, alpha=0.9, zorder=5)
        for widget in self.graph_frame.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self._canvas_ref = canvas

        self._update_waterfall(filtered, sr)
        wpm = self._estimate_wpm(events)
        self.wpm_value_label.configure(text=str(wpm) if wpm else "---")

        self._audio_duration = len(y) / sr
        self._stop_playback = False
        self.play_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self._smeter_active = True
        self._set_status("PLAYING", _GREEN)

        # raw_data already loaded in background — _play() has no I/O, timestamp is tight
        def _play():
            try:
                if raw_data is not None:
                    self._decode_start_time = time.time()
                    sd.play(raw_data, raw_sr)
                else:
                    data, samplerate = sf.read(self.audio_file)
                    self._decode_start_time = time.time()
                    sd.play(data, samplerate)
                sd.wait()
            finally:
                self.app.after(0, self._on_playback_done)

        self._playback_thread = threading.Thread(target=_play, daemon=True)
        self._playback_thread.start()

        # OFFSET_MS covers thread startup only (no file I/O in thread anymore)
        OFFSET_MS = 50
        self.app.after(OFFSET_MS, lambda s=session: self._update_playhead(s))

        for time_sec, etype, data in events:
            delay_ms = int(time_sec * 1000) + OFFSET_MS
            if etype == 'symbol':
                self.app.after(delay_ms, lambda s=session, sym=data: self._show_symbol(s, sym))
            elif etype == 'letter':
                self.app.after(delay_ms, lambda s=session, l=data: self._show_letter(s, l))
            elif etype == 'word':
                self.app.after(delay_ms, lambda s=session: self._show_word(s))

        if final_text:
            settle_ms = max((int(t * 1000) for t, _, _ in events), default=0) + OFFSET_MS + 300
            self.app.after(settle_ms, lambda s=session, txt=final_text: self._set_final_text(s, txt))

    def toggle_ai(self):
        if self.ai_visible:
            self.ai_frame.grid_remove()
            self.ai_btn.configure(text="◈ SHOW AI PREDICTION")
            self.ai_visible = False
        else:
            self.ai_frame.grid()
            self.ai_btn.configure(text="◈ HIDE AI PREDICTION")
            self.ai_visible = True

    def on_close(self):
        sd.stop()
        self.app.quit()
        self.app.destroy()

    def show(self):
        self.app.mainloop()
