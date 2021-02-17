import json
import logging
from multiprocessing import Lock
import os
import pickle
from typing import List, Dict, Optional, Any

from fluidml.storage import ResultsStore


logger = logging.getLogger(__name__)


class LocalFileStore(ResultsStore):
    def __init__(self, base_dir: str):
        super().__init__()
        self.base_dir = base_dir
        self._save_load_fn_from_type = {'json': (self._save_json, self._load_json),
                                        'pickle': (self._save_pickle, self._load_pickle)}
        self._lock = None

    # This is a workaround for Jupyter. Locks cannot be instantiated from within a Notebook.
    # Since the user instantiates the FileStore himself, the lock would be instantiated in a Notebook if the user
    # works there. Hence, we wrap it in a property such that the lock only gets instantiated at runtime from within
    # fluidml. This can be changed back once the Jupyter issue is fixed.
    @property
    def lock(self):
        if self._lock is None:
            self._lock = Lock()
        return self._lock

    @staticmethod
    def _save_json(name: str, obj: Dict, run_dir: str):
        json.dump(obj, open(os.path.join(run_dir, f'{name}.json'), "w"))

    @staticmethod
    def _load_json(name: str, run_dir: str) -> Dict:
        return json.load(open(os.path.join(run_dir, f'{name}.json'), "r"))

    @staticmethod
    def _save_pickle(name: str, obj: Any, run_dir: str):
        pickle.dump(obj, open(os.path.join(run_dir, f'{name}.p'), "wb"))

    @staticmethod
    def _load_pickle(name: str, run_dir: str) -> Any:
        return pickle.load(open(os.path.join(run_dir, f'{name}.p'), "rb"))

    def save(self, obj: Any, name: str, type_: str, task_name: str, task_unique_config: Dict, **kwargs):

        run_dir = self.get_context(task_name=task_name, task_unique_config=task_unique_config)

        # get save function for type
        save_fn, _ = self._save_load_fn_from_type[type_]

        # save object
        save_fn(name=name, obj=obj, run_dir=run_dir, **kwargs)

        # save load info
        load_info = {'kwargs': kwargs,
                     'type_': type_}
        pickle.dump(load_info, open(os.path.join(run_dir, f'.{name}_load_info.p'), 'wb'))

    def load(self, name: str, task_name: str, task_unique_config: Dict) -> Optional[Any]:
        task_dir = os.path.join(self.base_dir, task_name)

        # try to get existing run dir
        run_dir = self._get_run_dir(task_dir=task_dir, task_config=task_unique_config)
        if run_dir is None:
            return None

        # get load information from run dir
        try:
            load_info = pickle.load(open(os.path.join(run_dir, f".{name}_load_info.p"), "rb"))
        except FileNotFoundError:
            logger.warning(f'"{name}" could not be found in store. Task will be executed again.')
            return None

        # unpack load info
        type_ = load_info['type_']
        kwargs = load_info['kwargs']

        # get load function for type
        _, load_fn = self._save_load_fn_from_type[type_]

        # load the saved object from run dir
        obj = load_fn(name=name, run_dir=run_dir, **kwargs)
        return obj

    def get_context(self, task_name: str, task_unique_config: Dict):
        """ Method to get the current task's storage context.
        E.g. the current run directory in case of LocalFileStore.
        """
        task_dir = os.path.join(self.base_dir, task_name)

        # try to get existing run dir
        run_dir = self._get_run_dir(task_dir=task_dir, task_config=task_unique_config)

        # create new run dir if run dir did not exist
        if run_dir is None:
            with self.lock:
                run_dir = LocalFileStore._make_run_dir(task_dir=task_dir)
                json.dump(task_unique_config, open(os.path.join(run_dir, f'config.json'), 'w'))
        return run_dir

    @staticmethod
    def _scan_task_dir(task_dir: str) -> List[str]:
        os.makedirs(task_dir, exist_ok=True)
        exist_run_dirs = [os.path.join(task_dir, d.name)
                          for d in os.scandir(task_dir)
                          if d.is_dir() and d.name.isdigit()]
        return exist_run_dirs

    def _get_run_dir(self, task_dir: str, task_config: Dict) -> Optional[str]:
        with self.lock:
            exist_run_dirs = LocalFileStore._scan_task_dir(task_dir=task_dir)
            for exist_run_dir in exist_run_dirs:
                try:
                    exist_config = json.load(open(os.path.join(exist_run_dir, 'config.json'), 'r'))
                except FileNotFoundError:
                    continue

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
