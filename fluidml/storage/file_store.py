import json
import os
import pickle
from shutil import rmtree
from typing import List, Dict, Optional

from fluidml.storage import ResultsStore


class LocalFileStore(ResultsStore):
    def __init__(self, base_dir: str):
        self.base_dir = base_dir

    def get_results(self, task_name: str, unique_config: Dict) -> Optional[Dict]:  # Optional[Tuple[Dict, str]]
        task_dir = os.path.join(self.base_dir, task_name)

        exist_run_dirs = LocalFileStore._scan_task_dir(task_dir=task_dir)
        run_dir = LocalFileStore._get_run_dir(task_config=unique_config, exist_run_dirs=exist_run_dirs)
        if run_dir:
            result = pickle.load(open(os.path.join(run_dir, 'result.p'), 'rb'))
            # TODO: History log file?
            return result  # , run_dir
        return None

    def save_results(self, task_name: str, unique_config: Dict, results: Dict) -> str:
        task_dir = os.path.join(self.base_dir, task_name)
        run_dir = LocalFileStore._make_run_dir(task_dir=task_dir)

        pickle.dump(results, open(os.path.join(run_dir, 'result.p'), 'wb'))
        json.dump(unique_config, open(os.path.join(run_dir, 'config.json'), 'w'))
        return run_dir

    def update_results(self, task_name: str, unique_config: Dict, results: Dict) -> str:
        task_dir = os.path.join(self.base_dir, task_name)

        # get existing run dir
        exist_run_dirs = LocalFileStore._scan_task_dir(task_dir=task_dir)
        run_dir = LocalFileStore._get_run_dir(task_config=unique_config, exist_run_dirs=exist_run_dirs)

        if run_dir:
            # delete existing task results
            LocalFileStore._delete_dir_content(d=run_dir)
        else:
            run_dir = LocalFileStore._make_run_dir(task_dir=task_dir)

        # save new task results
        pickle.dump(results, open(os.path.join(run_dir, 'result.p'), 'wb'))
        json.dump(unique_config, open(os.path.join(run_dir, 'config.json'), 'w'))
        return run_dir

    @staticmethod
    def _scan_task_dir(task_dir: str) -> List[str]:
        os.makedirs(task_dir, exist_ok=True)
        exist_run_dirs = [os.path.join(task_dir, d.name)
                          for d in os.scandir(task_dir)
                          if d.is_dir() and d.name.isdigit()]
        return exist_run_dirs

    @staticmethod
    def _delete_dir_content(d: str):
        assert isinstance(d, str)
        for element in os.scandir(d):
            try:
                if element.is_file() or os.path.islink(element.path):
                    os.unlink(element.path)
                elif element.is_dir():
                    rmtree(element.path)
            except OSError as e:
                print(f'Failed to delete content in {element.path}. Reason: {e}.')

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
        exist_run_dirs = LocalFileStore._scan_task_dir(task_dir=task_dir)
        new_id = max([int(os.path.split(d)[-1]) for d in exist_run_dirs]) + 1 if exist_run_dirs else 0
        new_run_dir = os.path.join(task_dir, f'{str(new_id).zfill(3)}')
        os.makedirs(new_run_dir, exist_ok=True)
        return new_run_dir
