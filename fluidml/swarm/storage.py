from abc import ABC, abstractmethod
import json
import os
from typing import List, Dict, Optional, Tuple

from fluidml.common import Task


class ResultsStorage(ABC):
    @abstractmethod
    def construct(self):
        raise NotImplementedError

    @abstractmethod
    def get_results(self, task: Task) -> Optional[Dict]:
        """ Query method to get the results if they exist already """
        raise NotImplementedError

    @abstractmethod
    def save_results(self, task: Task, results: Dict):
        """ Method to save new results """
        raise NotImplementedError

    @abstractmethod
    def destruct(self):
        raise NotImplementedError


class LocalFileStorage(ResultsStorage):
    def __init__(self, base_dir: str):
        self.base_dir = base_dir

    def get_results(self, task: Task) -> Optional[Tuple[Dict, str]]:
        task_dir = os.path.join(self.base_dir, task.name)

        exist_run_dirs = LocalFileStorage._scan_task_dir(task_dir=task_dir)
        run_dir = LocalFileStorage._get_run_dir(task_config=task.unique_config, exist_run_dirs=exist_run_dirs)
        if run_dir:
            result = json.load(open(os.path.join(run_dir, 'result.json'), 'r'))
            return result, run_dir
        return None

    def save_results(self, task: Task, results: Dict) -> str:
        task_dir = os.path.join(self.base_dir, task.name)
        run_dir = LocalFileStorage._make_run_dir(task_dir=task_dir)

        json.dump(task.storage_path, open(os.path.join(run_dir, 'info.json'), 'w'))
        json.dump(results, open(os.path.join(run_dir, 'result.json'), 'w'))
        json.dump(task.unique_config, open(os.path.join(run_dir, 'config.json'), 'w'))
        return run_dir

    @staticmethod
    def _scan_task_dir(task_dir: str) -> List[str]:
        os.makedirs(task_dir, exist_ok=True)
        exist_run_dirs = [os.path.join(task_dir, d.name)
                          for d in os.scandir(task_dir)
                          if d.is_dir and d.name.isdigit()]
        return exist_run_dirs

    @staticmethod
    def _get_run_dir(task_config: Dict, exist_run_dirs: List[str]) -> Optional[str]:
        for exist_run_dir in exist_run_dirs:
            try:
                exist_config = json.load(open(os.path.join(exist_run_dir, 'config.json'), 'r'))
            except FileNotFoundError:
                continue
            if task_config == exist_config:
                return exist_run_dir
        return None

    @staticmethod
    def _make_run_dir(task_dir: str) -> str:
        exist_run_dirs = LocalFileStorage._scan_task_dir(task_dir=task_dir)
        new_id = max([int(os.path.split(d)[-1]) for d in exist_run_dirs]) + 1 if exist_run_dirs else 0
        new_run_dir = os.path.join(task_dir, f'{str(new_id).zfill(3)}')
        os.makedirs(new_run_dir, exist_ok=True)
        return new_run_dir

    def construct(self):
        raise ValueError('The LocalFileStorage needs no construction.')

    def destruct(self):
        raise ValueError('The LocalFileStorage needs no destruction.')
