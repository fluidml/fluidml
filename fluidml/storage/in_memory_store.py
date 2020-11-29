from fluildml.base.storage import ResultsStore
from multiprocessing import Manager
from typing import Dict, Optional


class InMemoryStore(ResultsStore):
    """
    This is an in-memory results store implemented using multiprocessing manager
    """
    def __init__(self):
        self._results_dict = Manager().dict()

    def save_results(self, task_name: str, unique_config: Dict, results: Dict):
        if task_name not in self._results_dict:
            self._results_dict[task_name] = []

        self._results_dict[task_name].append({
            "results": results,
            "config": unique_config
        })

    def get_results(self, task_name: str, unique_config: Dict) -> Optional[Dict]:
        if task_name not in self._results_dict:
            return None

        for result in self._results_dict[task_name]:
            if result["config"] == unique_config:
                return result["results"]

    def update_results(self, task_name: str, unique_config: Dict, results: Dict):
        # not required to delete the results, since it is in-memory
        # results will be cleared after each run and
        # a task would not run more than once
        self.save_results(self, task_name, unique_config)
