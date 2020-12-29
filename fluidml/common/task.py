from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional

from fluidml.common import DependencyMixin


@dataclass
class Resource:
    pass


class Task(ABC, DependencyMixin):
    """Abstract class for task"""

    def __init__(self,
                 kwargs: Optional[Dict] = None):
        DependencyMixin.__init__(self)
        self.name = None
        self.id_ = None
        self.kwargs = kwargs
        self.unique_config: Optional[Dict] = None
        self.force: Optional[bool] = None

    @abstractmethod
    def run(self,
            results: Dict[str, Any],
            resource: Resource) -> Optional[Dict[str, Any]]:
        """Implementation of core logic of task

        Args:
            results (Dict[str, Any]): results from predecessors (automatically passed by swarm)
            resource (Resource): resource to use (automatically passed by swarm)

        Returns:
            Optional[Dict[str, Any]]: a dict of results (automatically passed to successor tasks)
        """

        raise NotImplementedError
