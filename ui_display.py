import customtkinter as ctk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np

class UIDisplay:
    def __init__(self):
        self.audio_file = None
        self.app = ctk.CTk()
        self.app.title("Morse Code Decoder")
        self.app.geometry("900x700")
        # Title
        title = ctk.CTkLabel(
            self.app, 
            text="🎵 Morse Code Decoder",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title.pack(pady=10)

        # Buttons frame
        btn_frame = ctk.CTkFrame(self.app)
        btn_frame.pack(pady=10)

        # Load File button
        self.load_btn = ctk.CTkButton(
            btn_frame,
            text="📂 Load Audio File",
            command=self.load_file
        )
        self.load_btn.pack(side="left", padx=10)

        # Decode button
        self.decode_btn = ctk.CTkButton(
            btn_frame,
            text="▶ Decode",
            command=self.decode
        )
        self.decode_btn.pack(side="left", padx=10)

        # File label
        self.file_label = ctk.CTkLabel(
            self.app,
            text="No file loaded",
            font=ctk.CTkFont(size=12)
        )
        self.file_label.pack(pady=5)
        # Graph frame
        self.graph_frame = ctk.CTkFrame(self.app)
        self.graph_frame.pack(pady=10, fill="both", expand=True, padx=20)

        # Decoded text label
        self.result_label = ctk.CTkLabel(
            self.app,
            text="Decoded Text: ",
            font=ctk.CTkFont(size=16)
        )
        self.result_label.pack(pady=5)

        # Decoded text display box
        self.text_box = ctk.CTkTextbox(
            self.app,
            height=80,
            font=ctk.CTkFont(size=20, weight="bold")
        )
        self.text_box.pack(pady=5, fill="x", padx=20)
        self.app.protocol("WM_DELETE_WINDOW", self.on_close)
        
    def load_file(self):
        # Open file dialog
        file_path = filedialog.askopenfilename(
            filetypes=[("Audio Files", "*.wav *.mp3")]
        )
        if file_path:
            self.audio_file = file_path
            self.file_label.configure(
                text=f"Loaded: {file_path.split('/')[-1]}"
            )
    def on_close(self):
        self.app.quit()
        self.app.destroy()
            
    def decode(self):
        if not self.audio_file:
            self.file_label.configure(text="⚠️ Please load a file first!")
            return

        from audio_loader import AudioLoader
        from signal_filter import SignalFilter
        from signal_visualizer import SignalVisualizer
        from morse_decoder import MorseDecoder, MORSE_CODE_DICT

        # Load and process audio
        loader = AudioLoader(self.audio_file)
        y, sr = loader.load()

        filter = SignalFilter(y, sr)
        filtered_audio = filter.filter()

        # Show graph inside window
        visualizer = SignalVisualizer(filtered_audio, sr)
        fig = visualizer.get_figure()
        
        # Embed graph in UI
        for widget in self.graph_frame.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        # Decode and show text
        decoder = MorseDecoder(filtered_audio, sr, MORSE_CODE_DICT)
        morse_sequence = decoder.decode()
        decoded_text = ''.join(morse_sequence)

        # Update text box
        self.text_box.delete("1.0", "end")
        self.text_box.insert("1.0", decoded_text)
        
    def show(self):
        self.app.mainloop()