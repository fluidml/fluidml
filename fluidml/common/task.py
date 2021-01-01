from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

from fluidml.common import DependencyMixin
from fluidml.storage import ResultsStore

@dataclass
class Resource:
    pass


class Task(ABC, DependencyMixin):
    """Abstract class for task"""

    def __init__(self,
                 kwargs: Optional[Dict] = None):
        DependencyMixin.__init__(self)
        self.kwargs = kwargs
        self._name: Optional[str] = None
        self._id: Optional[int] = None
        self._unique_config: Optional[Dict] = None
        self._results_store: Optional[ResultsStore] = None
        self._resource: Optional[Resource] = None
        self._force: Optional[str] = None
        self._reduce = False

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name: str):
        self._name = name

    @property
    def id_(self):
        return self._id

    @id_.setter
    def id_(self, id: int):
        self._id = id

    @property
    def unique_config(self):
        return self._unique_config

    @unique_config.setter
    def unique_config(self, config: Dict):
        self._unique_config = config

    @property
    def results_store(self):
        return self._results_store

    @results_store.setter
    def results_store(self, results_store: ResultStore):
        self._results_store = results_store

    @property
    def resource(self):
        return self._resource

    @resource.setter
    def resource(self, resource: Resource):
        self._resource = resource

    @property
    def force(self):
        return self._force

    @force.setter
    def force(self, force: str):
        self._force = force

    @property
    def reduce(self):
        return self._reduce

    @reduce.setter
    def reduce(self, reduce: bool):
        self._reduce = reduce

    @abstractmethod
    def run(self, **results):
        """Implementation of core logic of task

        Args:
            results (Dict[str, Any]): results from predecessors (automatically passed by swarm)
        """

        raise NotImplementedError
