import numpy as np
import matplotlib.pyplot as plt

class SignalVisualizer:
    def __init__(self, y, sr):
        self.audio_data = y
        self.sampling_rate = sr
        
    def plot(self):
        plt.figure(figsize=(12, 6))
        time = np.linspace(0, len(self.audio_data) / self.sampling_rate, len(self.audio_data))
        plt.plot(time, self.audio_data)
        plt.title('Morse Code Signal')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid()
        plt.show()
        
    def get_figure(self):
        fig, ax = plt.subplots(figsize=(8, 3))
        time = np.linspace(0, len(self.audio_data) / self.sampling_rate, len(self.audio_data))
        ax.plot(time, self.audio_data, color='#C17F24', linewidth=0.8)
        ax.set_title('Morse Code Signal', color='#2C1A0E', fontsize=11)
        ax.set_xlabel('Time (s)', color='#8B6B4A', fontsize=9)
        ax.set_ylabel('Amplitude', color='#8B6B4A', fontsize=9)
        ax.grid(True, color='#E8D5B7', linewidth=0.6)
        fig.patch.set_facecolor('#FDF6EC')
        ax.set_facecolor('#FDF6EC')
        ax.tick_params(colors='#8B6B4A', labelsize=8)
        for spine in ax.spines.values():
            spine.set_color('#D4B896')
        fig.tight_layout(pad=1.5)
        return fig