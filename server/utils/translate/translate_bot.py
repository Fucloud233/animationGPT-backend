from abc import abstractstaticmethod

class TranslateBot:

    @abstractstaticmethod
    def check_connect():
        ...

    @abstractstaticmethod
    def translate(text: str):
        ...

    