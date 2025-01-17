from abc import ABC


class BaseEntity(ABC):
    def __init__(self, name=None):
        self.name = name
