import json
import os
from typing import Any, Dict, Callable, List

from busy_bee.common import Task, Resource
from busy_bee.common.logging import Console


class MyTask(Task):
    """A constructor class that creates a task object from a callable."""

    def __init__(self,
                 id_: int,
                 name: str,
                 task: Callable,
                 kwargs: Dict):
        super().__init__(id_=id_, name=name)
        self.task = task
        self.kwargs = kwargs
        self.config = None

    def run(self, results: Dict[str, Any], resource: Resource):
        task_dir = os.path.join(resource.base_dir, self.name)
        exist_run_dirs = MyTask._scan_task_dir(task_dir=task_dir)
        run_dir = self._get_run_dir(exist_run_dirs=exist_run_dirs)

        if run_dir:
            Console.get_instance().log(f'Task {self.name}-{self.id_} already executed.')
            result = json.load(open(os.path.join(run_dir, 'result.json'), 'r'))

        else:
            run_dir = MyTask._make_run_dir(task_dir=task_dir, exist_run_dirs=exist_run_dirs)
            result = self.task(results, resource, **self.kwargs)
            json.dump(result, open(os.path.join(run_dir, 'result.json'), 'w'))
            json.dump(self.config, open(os.path.join(run_dir, 'config.json'), 'w'))
        return result

    @staticmethod
    def _scan_task_dir(task_dir: str) -> List[str]:
        os.makedirs(task_dir, exist_ok=True)
        exist_run_dirs = [os.path.join(task_dir, d.name)
                          for d in os.scandir(task_dir)
                          if d.is_dir and d.name.isdigit()]
        return exist_run_dirs

    def _get_run_dir(self, exist_run_dirs: List[str]):
        for exist_run_dir in exist_run_dirs:
            try:
                exist_config = json.load(open(os.path.join(exist_run_dir, 'config.json'), 'r'))
            except FileNotFoundError:
                continue
            if self.config == exist_config:
                return exist_run_dir
        return None

    @staticmethod
    def _make_run_dir(task_dir: str, exist_run_dirs: List[str]) -> str:
        new_id = max([int(os.path.split(d)[-1]) for d in exist_run_dirs]) + 1 if exist_run_dirs else 0
        new_run_dir = os.path.join(task_dir, f'{str(new_id).zfill(3)}')
        os.makedirs(new_run_dir, exist_ok=True)
        return new_run_dir
