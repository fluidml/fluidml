from dataclasses import dataclass
import json
import logging
from multiprocessing import Lock
import os
import pickle
from typing import List, Dict, Optional, Any, Callable

from fluidml.storage import ResultsStore


logger = logging.getLogger(__name__)


@dataclass
class TypeInfo:
    save_fn: Callable  # save function used to save the object to store
    load_fn: Callable  # load function used to load the object from store
    extension: str     # file extension the object is saved with


class LocalFileStore(ResultsStore):
    def __init__(self, base_dir: str):
        super().__init__()
        self.base_dir = base_dir
        self._type_registry = {'json': TypeInfo(self._save_json, self._load_json, 'json'),
                               'pickle': TypeInfo(self._save_pickle, self._load_pickle, 'p')}
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
    def _save_json(name: str, obj: Dict, obj_dir: str, extension: str):
        json.dump(obj, open(os.path.join(obj_dir, f'{name}.{extension}'), "w"))

    @staticmethod
    def _load_json(name: str, obj_dir: str, extension: str) -> Dict:
        return json.load(open(os.path.join(obj_dir, f'{name}.{extension}'), "r"))

    @staticmethod
    def _save_pickle(name: str, obj: Any, obj_dir: str, extension: str):
        pickle.dump(obj, open(os.path.join(obj_dir, f'{name}.{extension}'), "wb"))

    @staticmethod
    def _load_pickle(name: str, obj_dir: str, extension: str) -> Any:
        return pickle.load(open(os.path.join(obj_dir, f'{name}.{extension}'), "rb"))

    def save(self,
             obj: Any,
             name: str,
             type_: str,
             task_name: str,
             task_unique_config: Dict,
             sub_dir: Optional[str] = None,
             **kwargs):

        run_dir = self.get_context(task_name=task_name, task_unique_config=task_unique_config)

        # get save function and file extension for type
        try:
            save_fn = self._type_registry[type_].save_fn
            extension = self._type_registry[type_].extension
        except KeyError:
            raise KeyError(f'Object type "{type_}" is not supported in {self.__class__.__name__}. Either extend it by '
                           f'implementing specific load and save functions for this type, or save the object as one '
                           f'of the following supported types: {", ".join(self._type_registry)}.')

        # set save directory for object
        obj_dir = run_dir
        if sub_dir is not None:
            obj_dir = os.path.join(run_dir, sub_dir)
            os.makedirs(obj_dir, exist_ok=True)

        # save object
        save_fn(name=name, obj=obj, obj_dir=obj_dir, extension=extension, **kwargs)

        # save load info
        load_info = {'kwargs': kwargs,
                     'obj_dir': os.path.relpath(obj_dir, run_dir),
                     'type_': type_}

        # create a hidden load info dir to save object information used by self.load()
        load_info_dir = os.path.join(run_dir, '.load_info')
        os.makedirs(load_info_dir, exist_ok=True)

        pickle.dump(load_info, open(os.path.join(load_info_dir, f'.{name}_load_info.p'), 'wb'))

    def load(self, name: str, task_name: str, task_unique_config: Dict, **kwargs) -> Optional[Any]:
        task_dir = os.path.join(self.base_dir, task_name)

        # try to get existing run dir
        run_dir = self._get_run_dir(task_dir=task_dir, task_config=task_unique_config)
        if run_dir is None:
            return None

        # get load information from run dir
        load_info_file_path = os.path.join(run_dir, '.load_info', f'.{name}_load_info.p')
        try:
            load_info = pickle.load(open(load_info_file_path, "rb"))
        except FileNotFoundError:
            logger.warning(f'"{name}" could not be found in store, since "{load_info_file_path}" does not exist.')
            return None

        # unpack load info
        type_ = load_info['type_']
        obj_dir = os.path.join(run_dir, load_info['obj_dir'])
        saved_kwargs = load_info['kwargs']

        # merge saved kwargs with user provided kwargs
        #  user provided kwargs overwrite saved kwargs when keys are identical
        merged_kwargs = {**saved_kwargs, **kwargs}

        # get load function for type
        load_fn = self._type_registry[type_].load_fn
        extension = self._type_registry[type_].extension

        # load the saved object from run dir
        try:
            obj = load_fn(name=name, obj_dir=obj_dir, extension=extension, **merged_kwargs)
        except FileNotFoundError:
            logger.warning(f'"{name}" could not be found in store. '
                           f'Note "{load_info_file_path}" does still exist.')
            return None
        return obj

    def delete(self, name: str, task_name: str, task_unique_config: Dict):
        task_dir = os.path.join(self.base_dir, task_name)

        # try to get existing run dir
        run_dir = self._get_run_dir(task_dir=task_dir, task_config=task_unique_config)
        if run_dir is None:
            logger.warning(f'"{name}" could not be deleted. '
                           f'No run directory for task "{task_name}" and the provided unique_config exists.')
            return None

        # get load information from run dir
        load_info_file_path = os.path.join(run_dir, '.load_info', f'.{name}_load_info.p')
        try:
            load_info = pickle.load(open(load_info_file_path, "rb"))
        except FileNotFoundError:
            logger.warning(f'"{name}" could not be deleted from store, since "{load_info_file_path}" does not exist. '
                           f'You might have to delete {name} manually.')
            return None

        # unpack load info
        type_ = load_info['type_']
        obj_dir = os.path.join(run_dir, load_info['obj_dir'])

        # use type_ to get the file extension
        extension = self._type_registry[type_].extension

        file_to_delete = os.path.join(obj_dir, f'{name}.{extension}')
        # remove the saved object and its load info file from the store
        try:
            os.remove(file_to_delete)
            os.remove(load_info_file_path)
        except FileNotFoundError:
            logger.warning(f'"{file_to_delete}" could not be deleted from store since it was not found.')

    def get_context(self, task_name: str, task_unique_config: Dict):
        """ Method to get the current task's storage context.
        E.g. the current run directory in case of LocalFileStore.
        """
        task_dir = os.path.join(self.base_dir, task_name)

        with self.lock:
            # try to get existing run dir
            run_dir = LocalFileStore._get_run_dir(task_dir=task_dir, task_config=task_unique_config)

            # create new run dir if run dir did not exist
            if run_dir is None:
                run_dir = LocalFileStore._make_run_dir(task_dir=task_dir)
                json.dump(task_unique_config, open(os.path.join(run_dir, f'config.json'), 'w'))
        return run_dir

    @staticmethod
    def _scan_task_dir(task_dir: str) -> List[str]:
        os.makedirs(task_dir, exist_ok=True)
        exist_run_dirs = [os.path.join(task_dir, d.name)
                          for d in os.scandir(task_dir)
                          if d.is_dir()]
        return exist_run_dirs

    @staticmethod
    def _get_run_dir(task_dir: str, task_config: Dict) -> Optional[str]:
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

        # get all numeric dir names in task dir and convert to ids
        ids = [int(d_name)
               for d_name in [os.path.split(d)[-1] for d in exist_run_dirs]
               if d_name.isdigit()] if exist_run_dirs else []

        # increment max id by 1 or start at 0
        new_id = max(ids) + 1 if ids else 0

        # create new run dir
        new_run_dir = os.path.join(task_dir, f'{str(new_id).zfill(3)}')
        os.makedirs(new_run_dir, exist_ok=True)
        return new_run_dir
