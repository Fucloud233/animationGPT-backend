from abc import abstractstaticmethod

class TranlsteBot:

    @abstractstaticmethod
    def check_connect():
        ...

    @abstractstaticmethod
    def translate(text: str):
        ...

    