from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

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
        # set in Task_spec
        self.kwargs = kwargs
        self._name: Optional[str] = None
        self._publishes: Optional[List[str]] = None
        self._expects: Optional[List[str]] = None

        # set in Flow
        self._id: Optional[int] = None
        self._unique_config: Optional[Dict] = None
        self._reduce = False
        self._force: Optional[str] = None

        # set in Dolphin
        self._results_store: Optional[ResultsStore] = None
        self._resource: Optional[Resource] = None

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
    def id_(self, id_: int):
        self._id = id_

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
    def results_store(self, results_store: ResultsStore):
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
    def publishes(self):
        return self._publishes

    @publishes.setter
    def publishes(self, publishes: List[str]):
        self._publishes = publishes

    @property
    def expects(self):
        return self._expects

    @expects.setter
    def expects(self, expects: List[str]):
        self._expects = expects

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

    def save(self, obj: Any, name: str, type_: Optional[str] = None, **kwargs):
        """Saves the given object to the results store

        Args:
            obj (Any): any object that is to be saved
            name (str): an unique name given to this object
            type_ (Optional[str], optional): additional type specification
                                             (eg. json, which is to be passed to results store). Defaults to None.
        """
        self.results_store.save(obj=obj, name=name, type_=type_,
                                task_name=self.name, task_unique_config=self.unique_config, **kwargs)

    def load(self, name: str, task_name: Optional[str] = None, task_unique_config: Optional[Dict] = None) -> Any:
        """Saves the given object to the results store

        Args:
            name (str): an unique name given to this object
            task_name (str): task name which saved the loaded object
            task_unique_config (Dict): unique config which specifies the run of the loaded object
        """
        if task_name is None and task_unique_config is None:
            obj = self.results_store.load(name=name, task_name=self.name, task_unique_config=self.unique_config)
        else:
            obj = self.results_store.load(name=name, task_name=task_name, task_unique_config=task_unique_config)
        return obj
