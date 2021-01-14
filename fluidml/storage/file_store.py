import json
import os
import pickle
from typing import List, Dict, Optional, Any

from fluidml.storage import ResultsStore


class LocalFileStore(ResultsStore):
    def __init__(self, base_dir: str):
        super().__init__()
        self.base_dir = base_dir
        self._save_load_fn_from_type = {'json': (self._save_json, self._load_json),
                                        'pickle': (self._save_pickle, self._load_pickle)}

    @staticmethod
    def _save_json(name: str, obj: Dict, run_dir: str):
        # save dict
        json.dump(obj, open(os.path.join(run_dir, f'{name}.json'), "w"))

    @staticmethod
    def _load_json(name: str, run_dir: str) -> Dict:
        return json.load(open(os.path.join(run_dir, f'{name}.json'), "r"))

    @staticmethod
    def _save_pickle(name: str, obj: Dict, run_dir: str):
        # save dict
        pickle.dump(obj, open(os.path.join(run_dir, f'{name}.p'), "wb"))

    @staticmethod
    def _load_pickle(name: str, run_dir: str) -> Dict:
        return pickle.load(open(os.path.join(run_dir, f'{name}.p'), "rb"))

    def save(self, obj: Any, name: str, type_: str, task_name: str, task_unique_config: Dict, **kwargs):
        task_dir = os.path.join(self.base_dir, task_name)

        # try to get existing run dir
        run_dir = LocalFileStore._get_run_dir(
            task_dir=task_dir, task_config=task_unique_config)

        # create new run dir if run dir did not exist
        if run_dir is None:
            run_dir = LocalFileStore._make_run_dir(task_dir=task_dir)
            json.dump(task_unique_config, open(os.path.join(run_dir, f'config.json'), "w"))

        # get save function for type
        save_fn, _ = self._save_load_fn_from_type[type_]

        # save object
        save_fn(name=name, obj=obj, run_dir=run_dir, **kwargs)

        # save load info
        load_info = {'kwargs': kwargs,
                     'type_': type_}
        pickle.dump(load_info, open(os.path.join(
            run_dir, f'.{name}_load_info.p'), 'wb'))

    def load(self, name: str, task_name: str, task_unique_config: Dict) -> Optional[Any]:
        task_dir = os.path.join(self.base_dir, task_name)

        # try to get existing run dir
        run_dir = LocalFileStore._get_run_dir(
            task_dir=task_dir, task_config=task_unique_config)
        if run_dir is None:
            return None

        # get load information from run dir
        try:
            load_info = pickle.load(
                open(os.path.join(run_dir, f".{name}_load_info.p"), "rb"))
        except FileNotFoundError:
            raise FileNotFoundError(f'{name} not saved.')

        # unpack load info
        type_ = load_info['type_']
        kwargs = load_info['kwargs']

        # get load function for type
        _, load_fn = self._save_load_fn_from_type[type_]

        # load the saved object from run dir
        obj = load_fn(name=name, run_dir=run_dir, **kwargs)
        return obj

    @staticmethod
    def _scan_task_dir(task_dir: str) -> List[str]:
        os.makedirs(task_dir, exist_ok=True)
        exist_run_dirs = [os.path.join(task_dir, d.name)
                          for d in os.scandir(task_dir)
                          if d.is_dir() and d.name.isdigit()]
        return exist_run_dirs

    @staticmethod
    def _get_run_dir(task_dir: str, task_config: Dict) -> Optional[str]:
        exist_run_dirs = LocalFileStore._scan_task_dir(task_dir=task_dir)
        for exist_run_dir in exist_run_dirs:
            try:
                exist_config = json.load(
                    open(os.path.join(exist_run_dir, 'config.json'), 'r'))
            except FileNotFoundError:
                continue
            # if task_config == exist_config:
            if exist_config.items() <= task_config.items():
                return exist_run_dir
        return None

    @staticmethod
    def _make_run_dir(task_dir: str) -> str:
        exist_run_dirs = LocalFileStore._scan_task_dir(task_dir=task_dir)
        new_id = max([int(os.path.split(d)[-1])
                      for d in exist_run_dirs]) + 1 if exist_run_dirs else 0
        new_run_dir = os.path.join(task_dir, f'{str(new_id).zfill(3)}')
        os.makedirs(new_run_dir, exist_ok=True)
        return new_run_dir
