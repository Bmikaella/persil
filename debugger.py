
class Debugger:

    def __init__(self, debug):
        self.debug=debug

    def print(self, text):
        if self.debug == 'True':
            print(f"DEBUG: {text}") 