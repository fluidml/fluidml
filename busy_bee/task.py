from abc import ABC, abstractmethod
from typing import List


class Task(ABC):
    """ Abstract class for tasks
    """
    def __init__(self, id_: int, pre_task_ids: List[int] = [],
                 post_task_ids: List[int] = []):
        self.id_ = id_
        self.pre_task_ids = pre_task_ids
        self.post_task_ids = post_task_ids

    @abstractmethod
    def run(self):
        """ Each concrete task must implement run() method
        """
        raise NotImplementedError
