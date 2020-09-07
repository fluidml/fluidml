from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass
from busy_bee.common.dependency import DependencyMixin


@dataclass(init=True)
class Resource:
    pass


class Task(ABC, DependencyMixin):
    """ Abstract class for task
    """
    def __init__(self, id: int, name: str):
        DependencyMixin.__init__(self)
        self.id_ = id
        self.name = name if name is not None else self.__class__.__name__

    @abstractmethod
    def run(self, results: Dict[str, Any], resource: Resource) -> Optional[Dict[str, Any]]:
        """
        Implementation of core logic of task

        Args:
            results (Dict[str, Any]): results from predecessors (automatically passed by swarm)
            resource (Resource): resource to use (automatically passed by swarm)

        Returns:
            Optional[Dict[str, Any]]: a dict of results (automatically passed to successor tasks)
        """
        raise NotImplementedError
