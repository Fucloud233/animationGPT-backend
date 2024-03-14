from datetime import datetime, timedelta

class Timer:
    def __init__(self):
        self.pre = None

    def start(self, text: str):
        self.pre = datetime.now()
        print(text)

    def tick(self, text: str):
        delta = datetime.now() - self.pre
        print(format_time(delta), " ", text)
        self.pre = datetime.now()


def format_time(timedelta: timedelta):
    return timedelta.seconds + timedelta.microseconds / 1e6