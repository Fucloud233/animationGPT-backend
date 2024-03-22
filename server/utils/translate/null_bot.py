from translate_bot import TranslateBot

class NullBot(TranslateBot):
    def check_connect():
        return False
    
    def translate(text: str):
        return (False, "")