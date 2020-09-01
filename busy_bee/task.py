from abc import ABC, abstractmethod
from typing import List, Optional


class Task(ABC):
    """ Abstract class for tasks
    """
    def __init__(self,
                 id_: int,
                 # name: Optional[str] = None,
                 pre_task_ids: Optional[List[int]] = None,
                 post_task_ids: Optional[List[int]] = None):
        if pre_task_ids is None:
            pre_task_ids = []
        if post_task_ids is None:
            post_task_ids = []

        # if name is None:
        #     name = self.__name__
        #
        # self.name = name
        self.id_ = id_
        self.pre_task_ids = pre_task_ids
        self.post_task_ids = post_task_ids

    @abstractmethod
    def run(self):
        """ Each concrete task must implement run() method
        """
        raise NotImplementedError
