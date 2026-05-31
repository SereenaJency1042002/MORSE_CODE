from ui_display import UIDisplay

class MorseApp:
    def __init__(self):
        pass
        
    def run(self):
        display = UIDisplay()
        display.show()

if __name__ == "__main__":
    app = MorseApp()
    app.run()