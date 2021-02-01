from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any


class ResultsStore(ABC):

    @abstractmethod
    def load(self, name: str, task_name: str, task_unique_config: Dict) -> Optional[Any]:
        """ Query method to load an object based on its name, task_name and task_config if it exists """
        raise NotImplementedError

    @abstractmethod
    def save(self, obj: Any, name: str, type_: str, task_name: str, task_unique_config: Dict, **kwargs):
        """ Method to save/update any artifact """
        raise NotImplementedError

    def get_results(self, task_name: str, task_unique_config: Dict, task_publishes: List[str]) -> Optional[Dict]:
        # if a task publishes no results, we always execute the task
        if not task_publishes:
            return None

        # here we loop over individual item names and call user provided self.load() to get individual item data
        results = {}
        for item_name in task_publishes:
            # load object
            obj: Optional[Any] = self.load(
                name=item_name, task_name=task_name, task_unique_config=task_unique_config)

            # if at least one expected result object of the task cannot be loaded, return None and re-run the task.
            if obj is None:
                return None

            # store object in results
            results[item_name] = obj

        return results
