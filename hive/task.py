from abc import ABC, abstractmethod


class Task(ABC):
    """Abstract class for tasks
    """
    def __init__(self, id: int):
        self.id = id

    @abstractmethod
    def run(self):
        """
        Each concrete task must implement run() method
        """
        raise NotImplementedError
