from typing import Optional, List, Union, Callable, Dict, Any
from abc import ABC, abstractmethod

from fluidml.common.dependency import DependencyMixin
from fluidml.common import Task
from fluidml.common.utils import MyTask


class BaseTaskSpec(DependencyMixin, ABC):
    def __init__(self,
                 task: Union[type, Callable],
                 name: Optional[str] = None):
        DependencyMixin.__init__(self)
        self.task = task
        self.name = name if name is not None else self.task.__name__

    def _create_task_object(self,
                            task_kwargs: Dict[str, Any],
                            task_id: Optional[int] = None) -> Task:
        if isinstance(self.task, type):
            task = self.task(id_=task_id, name=self.name, **task_kwargs)
            task.kwargs = task_kwargs
        elif isinstance(self.task, Callable):
            task = MyTask(id_=task_id, task=self.task, name=self.name, kwargs=task_kwargs)
        else:
            raise TypeError(f'{self.task} needs to be a Class object (type="type") or a Callable, e.g. a function.'
                            f'But it is of type "{type(self.task)}".')
        return task

    @abstractmethod
    def build(self) -> List[Task]:
        """Builds task from the specification

        Returns:
            List[Task]: task objects that are created
        """

        raise NotImplementedError
