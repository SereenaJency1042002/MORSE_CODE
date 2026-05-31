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
        ax.plot(time, self.audio_data, color='cyan')
        ax.set_title('Morse Code Signal')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.grid(True)
        fig.patch.set_facecolor('#2b2b2b')
        ax.set_facecolor('#2b2b2b')
        ax.tick_params(colors='white')
        ax.title.set_color('white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        return fig