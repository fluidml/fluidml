from abc import ABC, abstractmethod
from typing import Optional, Dict


class ResultsStore(ABC):
    @abstractmethod
    def get_results(self, task_name: str, unique_config: Dict) -> Optional[Dict]:
        """ Query method to get the results if they exist already """
        raise NotImplementedError

    @abstractmethod
    def save_results(self, task_name: str, unique_config: Dict, results: Dict):
        """ Method to save new results """
        raise NotImplementedError

    @abstractmethod
    def update_results(self, task_name: str, unique_config: Dict, results: Dict):
        """ Method to overwrite and update existing results """
        raise NotImplementedError
