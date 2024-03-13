from abc import abstractmethod

MAX_NUM  = 2000

class Node:
    def __init__(self, key, value, next=None, pre=None):
        self.key = key
        self.value = value
        self.pre: Node = pre
        self.next: Node = next

class Cache:
    @abstractmethod
    def add(self, key, value):
        pass

    def check(self, key):
        return self.add(key) != None
    
    @abstractmethod
    def get(self, key):
        pass

    def update(self, key, value):
        pass