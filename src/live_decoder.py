import os
import sys
import threading
import time
from datetime import datetime
import numpy as np
import sounddevice as sd
from src.signal_filter import SignalFilter
from src.morse_decoder import MorseDecoder, MORSE_CODE_DICT

SAMPLE_RATE = 16000
CHUNK_DURATION = 1.0
BUFFER_DURATION = 3.0
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)    # 16000
BUFFER_SAMPLES = int(SAMPLE_RATE * BUFFER_DURATION)  # 48000
CALIBRATION_CHUNKS = 5  # wait 5 seconds before first decode


class LiveDecoder:
    def __init__(self, device=None, sample_rate=16000,
                 on_text_callback=None, on_status_callback=None,
                 on_audio_callback=None):
        """
        device: sounddevice device index or name. None = system default.
        sample_rate: audio sample rate. Default 16000Hz.
        on_text_callback(new_text: str): called when new letters are decoded.
            Only the NEW part of the text is passed — not the full text.
            UI should append this to existing text box content.
        on_status_callback(status: str): called with status messages like
            "CALIBRATING 4s", "LIVE", "ERROR", "STOPPED"
        """
        self.device = device
        self.sample_rate = sample_rate
        self.on_text_callback = on_text_callback
        self.on_status_callback = on_status_callback
        self.on_audio_callback = on_audio_callback

        # Rolling buffer — always holds last 3 seconds of audio
        self._buffer = np.zeros(BUFFER_SAMPLES, dtype=np.float32)
        self._lock = threading.Lock()

        # State
        self._running = False
        self._stream = None
        self._process_thread = None
        self._chunks_collected = 0
        self._last_shown_text = ""  # tracks what was already sent to UI
        self._dot_threshold = None  # learned during calibration
        self._live_announced = False

        # Session logging/diagnostics state
        self._session_start    = None
        self._session_freq     = "unknown"
        self._anomaly_log      = []
        self._last_snr         = None
        self._last_peak_freq   = None
        self._chunk_count      = 0
        self._total_pulses     = 0
        self._total_questions  = 0
        self._total_chars      = 0
        self._best_period_start = None
        self._best_period_end   = None
        self._best_period_score = 0.0
        self._log_path         = None

    def start(self):
        """Start capturing and decoding live audio."""
        if self._running:
            return
        self._running = True
        self._chunks_collected = 0
        self._last_shown_text = ""
        self._live_announced = False
        # on_audio_callback already set in __init__, no reset needed
        self._buffer = np.zeros(BUFFER_SAMPLES, dtype=np.float32)

        # Reset session logging/diagnostics state
        self._session_start    = datetime.now()
        self._session_freq     = "unknown"
        self._anomaly_log      = []
        self._last_snr         = None
        self._last_peak_freq   = None
        self._chunk_count      = 0
        self._total_pulses     = 0
        self._total_questions  = 0
        self._total_chars      = 0
        self._best_period_start = None
        self._best_period_end   = None
        self._best_period_score = 0.0
        self._log_path         = None

        # Open sounddevice input stream
        # blocksize=CHUNK_SAMPLES means callback fires exactly every 1 second
        self._stream = sd.InputStream(
            device=self.device,
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32',
            blocksize=CHUNK_SAMPLES,
            callback=self._audio_callback,
        )
        self._stream.start()

        # Start background processing thread
        self._process_thread = threading.Thread(
            target=self._process_loop, daemon=True
        )
        self._process_thread.start()

        if self.on_status_callback:
            self.on_status_callback(f"CALIBRATING {CALIBRATION_CHUNKS}s")

    def stop(self):
        """Stop capturing and decoding."""
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._process_thread:
            self._process_thread.join(timeout=3.0)
            self._process_thread = None
        self._last_shown_text = ""
        self._dot_threshold = None
        self._live_announced = False
        self.print_session_report()
        if self.on_status_callback:
            self.on_status_callback("STOPPED")

    def set_frequency(self, freq_str: str):
        """Called from UI when user sets frequency."""
        self._session_freq = freq_str.strip()
        if self._session_start and not self._log_path:
            ts = self._session_start.strftime("%Y-%m-%d_%H-%M-%S")
            clean_freq = self._session_freq.replace('.', '')
            self._log_path = os.path.join(
                "recordings",
                f"session_{ts}_{clean_freq}kHz.log"
            )

    def _elapsed(self) -> str:
        """Return elapsed time as MM:SS string."""
        if not self._session_start:
            return "00:00"
        secs = int((datetime.now() - self._session_start).total_seconds())
        return f"{secs//60:02d}:{secs%60:02d}"

    def _estimate_snr(self, audio: np.ndarray) -> float:
        """Estimate SNR from audio buffer."""
        rms = float(np.sqrt(np.mean(audio ** 2)))
        noise_floor = float(np.percentile(np.abs(audio), 10))
        if noise_floor < 1e-9:
            return 99.0
        return round(20 * np.log10(rms / (noise_floor + 1e-9)), 1)

    def _estimate_peak_freq(self, audio: np.ndarray) -> float:
        """Estimate dominant frequency in audio."""
        fft   = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1 / self.sample_rate)
        mask  = (freqs >= 200) & (freqs <= 5000)
        if not np.any(mask):
            return 0.0
        return round(float(freqs[mask][np.argmax(fft[mask])]), 1)

    def _log_anomaly(self, tag: str, message: str, metrics: dict):
        """Write anomaly to log file and memory."""
        entry = {
            "time":    self._elapsed(),
            "tag":     tag,
            "message": message,
            "metrics": metrics,
        }
        self._anomaly_log.append(entry)

        if self._log_path:
            os.makedirs("recordings", exist_ok=True)
            with open(self._log_path, 'a', encoding='utf-8') as f:
                f.write(f"\n[t={entry['time']}] {tag} — {message}\n")
                for k, v in metrics.items():
                    f.write(f"          {k}: {v}\n")

    def _print_live_status(self, snr: float, peak_freq: float,
                            dot_ms: float, dash_ms: float,
                            ratio: float, q_rate: float,
                            status: str):
        """Print single updating status line to terminal."""
        line = (
            f"[LIVE t={self._elapsed()}] "
            f"{peak_freq:.0f}Hz | "
            f"dot={dot_ms:.0f}ms dash={dash_ms:.0f}ms | "
            f"ratio={ratio:.2f}x | "
            f"SNR={snr:.1f}dB | "
            f"?={q_rate:.0%} | "
            f"{status}"
        )
        sys.stdout.write("\r" + line.ljust(110))
        sys.stdout.flush()

    def print_session_report(self):
        """Print full session summary — called when STOP is clicked."""
        if not self._session_start:
            return
        duration = int(
            (datetime.now() - self._session_start).total_seconds()
        )
        mins, secs = duration // 60, duration % 60

        q_rate = (self._total_questions / self._total_chars
                  if self._total_chars > 0 else 0.0)

        print("\n")
        print("━" * 50)
        print(f"SESSION REPORT — {mins}m {secs}s")
        print("━" * 50)
        print(f"Frequency:       {self._session_freq} kHz")
        print(f"Total chunks:    {self._chunk_count}")
        print(f"Total pulses:    {self._total_pulses}")
        print(f"Confusion rate:  {q_rate:.1%}")
        print(f"Anomalies logged: {len(self._anomaly_log)}")
        if self._best_period_start:
            print(f"Best period:     "
                  f"t={self._best_period_start} → t={self._best_period_end}")
        if self._log_path and os.path.exists(self._log_path):
            print(f"Log saved:       {self._log_path}")
        print("━" * 50)

        if self._anomaly_log:
            print(f"  → Full anomaly log: {self._log_path}")
        print()

    def _analyse_chunk(self, audio: np.ndarray, decoded_text: str):
        """
        Analyse each decoded chunk.
        Updates live status line.
        Logs anomalies when thresholds exceeded.
        Tracks session statistics.
        """
        self._chunk_count += 1

        # Estimate signal metrics
        snr       = self._estimate_snr(audio)
        peak_freq = self._estimate_peak_freq(audio)

        # Count characters and ? marks
        chars     = len(decoded_text.replace(' ', ''))
        questions = decoded_text.count('?')
        self._total_chars     += chars
        self._total_questions += questions
        q_rate = questions / chars if chars > 0 else 0.0

        # Estimate dot/dash from audio
        on_durs = []
        try:
            from src.signal_filter import SignalFilter
            filtered = SignalFilter(audio, self.sample_rate).filter()
            frame_length = max(64, int(self.sample_rate * 0.01))
            hop_length   = frame_length // 2
            import librosa
            rms = librosa.feature.rms(
                y=filtered,
                frame_length=frame_length,
                hop_length=hop_length
            )[0]
            noise_floor = np.max(rms) * 0.05
            active_rms  = rms[rms > noise_floor]
            threshold   = np.median(active_rms) * 0.6 if len(active_rms) > 0 else 0
            signal_on   = rms > threshold
            on_durs = []
            count = 0
            for val in signal_on:
                if val:
                    count += 1
                else:
                    if count > 0:
                        on_durs.append((count * hop_length) / self.sample_rate * 1000)
                        count = 0
            if count > 0:
                on_durs.append((count * hop_length) / self.sample_rate * 1000)

            if len(on_durs) >= 2:
                dot_ms  = float(np.percentile(on_durs, 30))
                dash_ms = float(np.percentile(on_durs, 70))
                ratio   = dash_ms / (dot_ms + 1e-9)
                self._total_pulses += len(on_durs)
            else:
                dot_ms = dash_ms = ratio = 0.0
        except Exception:
            dot_ms = dash_ms = ratio = 0.0

        # Determine status and log anomalies
        status   = "✅ STABLE"
        metrics  = {
            "SNR":       f"{snr:.1f}dB",
            "freq":      f"{peak_freq:.0f}Hz",
            "dot":       f"{dot_ms:.0f}ms",
            "dash":      f"{dash_ms:.0f}ms",
            "ratio":     f"{ratio:.2f}x",
            "confusion": f"{q_rate:.1%}",
        }

        if not on_durs:
            status = "🔴 NO SIGNAL"
            self._log_anomaly("🔴 NO SIGNAL",
                              "No pulses detected — transmission gap or station off",
                              metrics)
        elif q_rate > 0.50:
            status = "🔴 HIGH QRM"
            self._log_anomaly("🔴 HIGH QRM",
                              "50%+ confusion — multiple stations or heavy interference",
                              metrics)
        elif q_rate > 0.25:
            status = "⚠️  CONFUSION"
            self._log_anomaly("⚠️  HIGH CONFUSION",
                              "25%+ unrecognized — weak signal or timing errors",
                              metrics)
        elif ratio > 0 and (ratio < 2.0 or ratio > 4.0):
            status = "⚠️  TIMING"
            self._log_anomaly("⚠️  TIMING UNSTABLE",
                              f"ratio={ratio:.2f}x outside normal range 2.0-4.0x",
                              metrics)
        elif snr < 8.0:
            status = "⚠️  WEAK"
            self._log_anomaly("⚠️  WEAK SIGNAL",
                              f"SNR={snr:.1f}dB below threshold — consider retuning",
                              metrics)

        # Track frequency drift
        if self._last_peak_freq and peak_freq > 0:
            drift = abs(peak_freq - self._last_peak_freq)
            if drift > 50:
                status = "⚠️  DRIFT"
                self._log_anomaly("⚠️  FREQUENCY DRIFT",
                                  f"{self._last_peak_freq:.0f}Hz → {peak_freq:.0f}Hz",
                                  metrics)
        self._last_peak_freq = peak_freq

        # Track best period (lowest confusion)
        if chars > 0 and q_rate < self._best_period_score or self._best_period_start is None:
            if q_rate < (self._best_period_score if self._best_period_start else 1.0):
                self._best_period_score = q_rate
                self._best_period_start = self._elapsed()
                self._best_period_end   = self._elapsed()
            elif self._best_period_start:
                self._best_period_end = self._elapsed()

        # Print live status line
        self._print_live_status(
            snr, peak_freq, dot_ms, dash_ms, ratio, q_rate, status
        )

    def _audio_callback(self, indata, frames, time_info, status):
        """
        Called by sounddevice every 1 second with new audio chunk.
        MUST be fast — no heavy processing here.

        indata shape: (CHUNK_SAMPLES, 1) — mono float32
        """
        if status:
            print(f"LiveDecoder audio status: {status}")

        # Take mono channel, flatten to 1D
        chunk = indata[:, 0].copy()

        # Roll buffer left by chunk size (drop oldest second)
        # Add new chunk at the end (newest second)
        with self._lock:
            self._buffer = np.roll(self._buffer, -CHUNK_SAMPLES)
            self._buffer[-CHUNK_SAMPLES:] = chunk
            self._chunks_collected += 1

    def _process_loop(self):
        """
        Background thread — runs every 1 second.
        Waits for calibration, then decodes buffer and finds new text.
        """
        while self._running:
            time.sleep(CHUNK_DURATION)

            if not self._running:
                break

            chunks = self._chunks_collected

            # Calibration phase — wait for enough audio
            if chunks < CALIBRATION_CHUNKS:
                remaining = CALIBRATION_CHUNKS - chunks
                if self.on_status_callback:
                    self.on_status_callback(f"CALIBRATING {remaining}s")
                continue

            # Copy buffer safely — reused for calibration (if needed) and decode
            with self._lock:
                audio_copy = self._buffer.copy()

            # Calibrate once at end of calibration phase
            if self._dot_threshold is None:
                self._dot_threshold = self._calibrate(audio_copy)
                if self.on_status_callback:
                    if self._dot_threshold:
                        self.on_status_callback("LIVE")
                    else:
                        self.on_status_callback("LIVE — NO SIGNAL DETECTED")
                self._live_announced = True
            elif not self._live_announced:
                if self.on_status_callback:
                    self.on_status_callback("LIVE")
                self._live_announced = True

            # Decode the buffer
            new_full_text = self._decode_buffer(audio_copy)

            # Send audio buffer to UI for visualization
            if self.on_audio_callback:
                self.on_audio_callback(audio_copy, self.sample_rate)

            if not new_full_text:
                continue

            # Find only the NEW part compared to what was already shown
            # Simple approach: if new text starts with last shown text,
            # the new part is everything after last shown text
            if new_full_text.startswith(self._last_shown_text):
                new_part = new_full_text[len(self._last_shown_text):]
            else:
                # Decoder produced different output (e.g. re-calibrated)
                # Reset and show full new text
                new_part = new_full_text
                self._last_shown_text = ""

            # Analyse chunk and update terminal
            self._analyse_chunk(audio_copy, new_full_text)

            # Only send to UI if there is actually something new
            if new_part.strip():
                self._last_shown_text = new_full_text
                if self.on_text_callback:
                    self.on_text_callback(new_part)

    def _calibrate(self, audio: np.ndarray) -> float:
        """
        Learn dot duration from calibration audio.
        Runs SignalFilter + RMS to find all ON pulse durations.
        Returns dot_threshold (30th percentile of short pulses).
        """
        try:
            from src.signal_filter import SignalFilter
            import librosa

            filtered = SignalFilter(audio, self.sample_rate).filter()

            frame_length = max(64, int(self.sample_rate * 0.01))
            hop_length = frame_length // 2
            rms = librosa.feature.rms(
                y=filtered,
                frame_length=frame_length,
                hop_length=hop_length
            )[0]

            noise_floor = np.max(rms) * 0.05
            active_rms = rms[rms > noise_floor]
            if len(active_rms) == 0:
                return None

            threshold = np.median(active_rms) * 0.6
            signal_on = rms > threshold

            # Find ON segment durations
            on_durations = []
            count = 0
            for val in signal_on:
                if val:
                    count += 1
                else:
                    if count > 0:
                        on_durations.append(count)
                        count = 0
            if count > 0:
                on_durations.append(count)

            if len(on_durations) < 3:
                return None

            on_durations = np.array(on_durations)

            # Remove noise spikes
            min_dur = np.mean(on_durations) * 0.3
            on_durations = on_durations[on_durations >= min_dur]

            if len(on_durations) < 2:
                return None

            # Bottom 30% = dots
            dot_threshold = float(np.percentile(on_durations, 30))
            print(f"[CALIBRATION] dot_threshold={dot_threshold:.2f} frames, "
                  f"total pulses={len(on_durations)}, "
                  f"mean={np.mean(on_durations):.2f}, "
                  f"min={np.min(on_durations):.2f}, "
                  f"max={np.max(on_durations):.2f}")
            return dot_threshold

        except Exception as e:
            print(f"Calibration error: {e}")
            return None

    def _decode_buffer(self, audio: np.ndarray) -> str:
        """
        Run full decode pipeline on audio buffer.
        Returns decoded text string or empty string on failure.
        """
        try:
            filtered = SignalFilter(audio, self.sample_rate).filter()
            _, text = MorseDecoder(
                filtered, self.sample_rate, MORSE_CODE_DICT,
                mode='adaptive' if self._dot_threshold else 'kmeans',
                dot_threshold=self._dot_threshold,
            ).decode_with_timing()
            return text.strip()
        except Exception as e:
            print(f"LiveDecoder decode error: {e}")
            return ""
