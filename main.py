import os
from dotenv import load_dotenv
from src.ui_display import UIDisplay

load_dotenv()


class MorseApp:
    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY") or None

    def run(self):
        display = UIDisplay(groq_api_key=self.groq_api_key)
        display.show()


if __name__ == "__main__":
    app = MorseApp()
    app.run()