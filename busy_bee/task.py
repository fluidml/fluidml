from abc import ABC, abstractmethod


class Task(ABC):
    """ Abstract class for tasks
    """
    def __init__(self, id_: int):
        self.id_ = id_

    @abstractmethod
    def run(self):
        """ Each concrete task must implement run() method
        """
        raise NotImplementedError
