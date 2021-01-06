from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any


class ResultsStore(ABC):
    def __init__(self):
        self._task_name: Optional[str] = None
        self._task_unique_config: Optional[Dict] = None
        self._task_publishes: Optional[List[str]] = None

    @property
    def task_name(self):
        return self._task_name

    @task_name.setter
    def task_name(self, task_name: str):
        self._task_name = task_name

    @property
    def task_unique_config(self):
        return self._task_unique_config

    @task_unique_config.setter
    def task_unique_config(self, task_unique_config: Dict):
        self._task_unique_config = task_unique_config

    @property
    def task_publishes(self):
        return self._task_publishes

    @task_publishes.setter
    def task_publishes(self, task_publishes: List[str]):
        self._task_publishes = task_publishes

    @abstractmethod
    def load(self, name: str) -> Optional[Any]:
        """ Query method to load an object based on its name, task_name and task_config if it exists """
        raise NotImplementedError

    @abstractmethod
    def save(self, obj: Any, name: str, type_: str, **kwargs):
        """ Method to save/update any artifact """
        raise NotImplementedError

    def get_results(self) -> Optional[Dict]:
        # here we loop over individual item names and call user provided self.load() to get individual item data
        results = {}
        for item_name in self.task_publishes:
            # load object
            obj: Optional[Any] = self.load(name=item_name)

            # if at least one expected result object of the task cannot be loaded, return None and re-run the task.
            if obj is None:
                return None

            # store object in results
            results[item_name] = obj
        return results
