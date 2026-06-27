import customtkinter as ctk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time
import sounddevice as sd
import soundfile as sf

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("dark-blue")

# Brown palette
_BG        = "#F2E4CC"   # warm parchment
_SURFACE   = "#FDF6EC"   # lighter cream (graph, textbox)
_BORDER    = "#D4B896"   # tan border
_TEXT      = "#2C1A0E"   # dark espresso
_MUTED     = "#8B6B4A"   # medium brown
_BUTTON    = "#6B3A2A"   # mahogany (all buttons same color)
_BTN_HOVER = "#5A2F22"   # darker mahogany
_AMBER     = "#C17F24"   # amber (morse symbols)


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

        self.app = ctk.CTk()
        self.app.title("Morse Code Decoder")
        self.app.geometry("940x780")
        self.app.configure(fg_color=_BG)

        self._build_ui()
        self.app.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self):
        self._loading = False
        # ── Title ─────────────────────────────────────────────────────────────
        ctk.CTkLabel(
            self.app,
            text="Morse Code Decoder",
            font=ctk.CTkFont(size=26, weight="bold"),
            text_color=_TEXT,
        ).pack(pady=(20, 4))

        ctk.CTkLabel(
            self.app,
            text="Load an audio file and decode its Morse signal",
            font=ctk.CTkFont(size=12),
            text_color=_MUTED,
        ).pack(pady=(0, 12))

        # ── Buttons (all same mahogany color) ─────────────────────────────────
        btn_frame = ctk.CTkFrame(self.app, fg_color="transparent")
        btn_frame.pack(pady=(0, 8))

        btn_style = dict(
            fg_color=_BUTTON, hover_color=_BTN_HOVER,
            font=ctk.CTkFont(size=13, weight="bold"),
            corner_radius=8, height=36,
        )

        self.load_btn = ctk.CTkButton(
            btn_frame, text="📂  Load File", width=140,
            command=self.load_file, **btn_style
        )
        self.load_btn.pack(side="left", padx=6)

        self.decode_btn = ctk.CTkButton(
            btn_frame, text="▶  Decode & Play", width=160,
            command=self.decode, **btn_style
        )
        self.decode_btn.pack(side="left", padx=6)

        self.play_btn = ctk.CTkButton(
            btn_frame, text="🔊  Play Audio", width=140,
            state="disabled", command=self.play_audio, **btn_style
        )
        self.play_btn.pack(side="left", padx=6)

        self.stop_btn = ctk.CTkButton(
            btn_frame, text="⏹  Stop", width=100,
            state="disabled", command=self.stop_audio, **btn_style
        )
        self.stop_btn.pack(side="left", padx=6)

        # ── File status ───────────────────────────────────────────────────────
        self.file_label = ctk.CTkLabel(
            self.app, text="No file loaded",
            font=ctk.CTkFont(size=11), text_color=_MUTED,
        )
        self.file_label.pack(pady=(2, 6))

        # ── Graph frame ───────────────────────────────────────────────────────
        self.graph_frame = ctk.CTkFrame(
            self.app, fg_color=_SURFACE,
            border_color=_BORDER, border_width=1, corner_radius=10,
        )
        self.graph_frame.pack(padx=30, pady=6, fill="both", expand=True)

        # ── Live morse label (amber, bold) ────────────────────────────────────
        self.morse_label = ctk.CTkLabel(
            self.app, text="",
            font=ctk.CTkFont(size=28, family="Courier New", weight="bold"),
            text_color=_AMBER,
        )
        self.morse_label.pack(pady=(10, 2))

        # ── Decoded text (raw) ────────────────────────────────────────────────
        ctk.CTkLabel(
            self.app, text="Decoded Text (Raw)",
            font=ctk.CTkFont(size=12, weight="bold"), text_color=_MUTED,
        ).pack(pady=(4, 2))

        self.text_box = ctk.CTkTextbox(
            self.app, height=75,
            fg_color=_SURFACE, text_color=_TEXT,
            border_color=_BORDER, border_width=1,
            corner_radius=8,
            font=ctk.CTkFont(size=22, weight="bold"),
        )
        self.text_box.pack(padx=30, pady=(0, 8), fill="x")

        # ── AI corrected text ─────────────────────────────────────────────────
        ctk.CTkLabel(
            self.app, text="AI Corrected Text (Groq)",
            font=ctk.CTkFont(size=12, weight="bold"), text_color=_MUTED,
        ).pack(pady=(4, 2))

        self.ai_text_box = ctk.CTkTextbox(
            self.app, height=75,
            fg_color=_SURFACE, text_color=_TEXT,
            border_color=_BORDER, border_width=1,
            corner_radius=8,
            font=ctk.CTkFont(size=22, weight="bold"),
        )
        self.ai_text_box.pack(padx=30, pady=(0, 20), fill="x")

    # ── file loading ──────────────────────────────────────────────────────────

    def load_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Audio Files", "*.wav *.mp3")]
        )
        if file_path:
            self.audio_file = file_path
            name = file_path.replace("\\", "/").split("/")[-1]
            self.file_label.configure(text=f"●  {name}", text_color=_BUTTON)
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
            self.app.after(50, lambda s=session: self._update_playhead(s))

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
            self.file_label.configure(text="⚠  Please load a file first!", text_color=_BUTTON)
            return

        self._cancel_animation()
        sd.stop()
        session = self._decode_session

        # Show spinner immediately while processing runs in background
        self.decode_btn.configure(state="disabled", text="⏳  Processing...")
        self.text_box.delete("1.0", "end")
        self.ai_text_box.delete("1.0", "end")
        self._start_spinner(session)

        def _process():
            from src.audio_loader import AudioLoader
            from src.signal_filter import SignalFilter
            from src.morse_decoder import MorseDecoder, MORSE_CODE_DICT
            from src.intelligent_corrector import IntelligentCorrector
            from src.groq_corrector import GroqCorrector

            y, sr = AudioLoader(self.audio_file).load()
            filtered = SignalFilter(y, sr).filter()
            events, raw_text = MorseDecoder(filtered, sr, MORSE_CODE_DICT).decode_with_timing()

            corrected_events, corrected_text = IntelligentCorrector().correct(events, raw_text)

            raw_morse = ' '.join(d for _, et, d in corrected_events if et == 'symbol')

            # Groq correction temporarily disabled
            # if self.groq_api_key:
            #     final_text = GroqCorrector(self.groq_api_key).correct(corrected_text, raw_morse)
            # else:
            #     final_text = corrected_text
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
        self.decode_btn.configure(state="normal", text="▶  Decode & Play")

        # Create figure on main thread (matplotlib requires this)
        from src.signal_visualizer import SignalVisualizer
        fig = SignalVisualizer(filtered, sr).get_figure()

        # Embed graph
        ax = fig.axes[0]
        self._playhead = ax.axvline(x=0, color=_BUTTON, linewidth=2, alpha=0.9, zorder=5)
        for widget in self.graph_frame.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self._canvas_ref = canvas

        self._audio_duration = len(y) / sr
        self._stop_playback = False
        self.play_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")

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

    def on_close(self):
        sd.stop()
        self.app.quit()
        self.app.destroy()

    def show(self):
        self.app.mainloop()
