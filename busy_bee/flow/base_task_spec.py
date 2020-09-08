from typing import Optional, List, Union, Callable, Dict, Any
from abc import ABC, abstractmethod

from busy_bee.common.dependency import DependencyMixin
from busy_bee.common import Task
from busy_bee.common.utils import MyTask


class BaseTaskSpec(DependencyMixin, ABC):
    def __init__(self, task: Union[type, Callable], name: Optional[str] = None):
        DependencyMixin.__init__(self)
        self.task = task
        self.name = name if name is not None else self.task.__name__

    def _type_to_task(self, task_id: int, task_args: Dict[str, Any]) -> Task:
        if isinstance(self.task, type):
            task = self.task(id_=task_id, name=self.name, **task_args)
        elif isinstance(self.task, Callable):
            task = MyTask(id_=task_id, task=self.task, name=self.name, kwargs=task_args)
        else:
            raise ValueError
        return task

    @abstractmethod
    def build(self) -> List[Task]:
        """
        Builds task from the specification

        Returns:
            List[Task]: task objects that are created
        """
        raise NotImplementedError
