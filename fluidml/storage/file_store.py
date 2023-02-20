import contextlib
import json
import logging
import os
import pickle
import shutil
from dataclasses import dataclass
from multiprocessing import Lock
from typing import List, Dict, Optional, Any, Callable, AnyStr, Tuple, IO, TYPE_CHECKING

from fluidml.storage.base import ResultsStore, Promise

if TYPE_CHECKING:
    from fluidml.common.task import RunInfo

logger = logging.getLogger(__name__)


class FilePromise(Promise):
    def __init__(
        self,
        name: str,
        path: str,
        save_fn: Callable,
        load_fn: Callable,
        open_fn: Optional[Callable] = None,
        mode: Optional[str] = None,
        load_kwargs: Optional[Dict] = None,
        **open_kwargs,
    ):
        super().__init__()

        self.name = name
        self.path = path
        self.save_fn = save_fn
        self.load_fn = load_fn
        self.open_fn = open_fn
        self.mode = mode
        self.load_kwargs = load_kwargs if load_kwargs is not None else {}
        self.open_kwargs = open_kwargs

    def load(self, **kwargs):
        kwargs = {**self.load_kwargs, **kwargs}
        try:
            if self.mode is None:
                return self.load_fn(self.path, **kwargs)
            else:
                with File(
                    self.path,
                    self.mode,
                    load_fn=self.load_fn,
                    save_fn=self.save_fn,
                    open_fn=self.open_fn,
                    **self.open_kwargs,
                ) as file:
                    return file.load(**kwargs)
        except FileNotFoundError:
            logger.warning(f'"{self.name}" could not be found in store.')
            return None
        except IsADirectoryError:
            return self.load_fn(self.path, **kwargs)


class File:
    def __init__(
        self,
        path: str,
        mode: str,
        save_fn: Callable,
        load_fn: Callable,
        open_fn: Optional[Callable] = None,
        load_kwargs: Optional[Dict] = None,
        **open_kwargs,
    ):
        self._path = path
        self._mode = mode
        self._save_fn = save_fn
        self._load_fn = load_fn
        self._open_kwargs = open_kwargs
        self._load_kwargs = load_kwargs if load_kwargs is not None else {}

        if open_fn is None:
            self.f = open(self._path, self._mode, **open_kwargs)
        else:
            self.f = open_fn(self._path, self._mode, **open_kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __iter__(self):
        return self.f.__iter__()

    def __next__(self):
        return self.f.__next__()

    def close(self):
        self.f.close()

    @property
    def closed(self):
        return self.f.closed

    def flush(self):
        self.f.flush()

    def readline(self, size: int = -1):
        return self.f.readline(size)

    def readlines(self, hint: int = -1):
        return self.f.readlines(hint)

    def readable(self):
        return self.f.readable()

    def writable(self):
        return self.f.writable()

    def seekable(self):
        return self.f.seekable()

    def seek(self, offset: int, whence=0):
        return self.f.seek(offset, whence)

    def tell(self):
        return self.f.tell()

    def truncate(self, size: Optional[int] = None):
        return self.f.truncate(size)

    def writelines(self, lines: List):
        return self.f.writelines(lines)

    def read(self, size: int = -1):
        self.f.read(size)

    def write(self, obj: AnyStr):
        self.f.write(obj)

    def load(self, **kwargs):
        kwargs = {**self._load_kwargs, **kwargs}
        return self._load_fn(self.f, **kwargs)

    def save(self, obj, **kwargs):
        self._save_fn(obj, self.f, **kwargs)

    @classmethod
    def from_promise(cls, promise: FilePromise):
        return cls(promise.path, promise.mode, promise.save_fn, promise.load_fn, promise.open_fn, **promise.open_kwargs)


@dataclass
class TypeInfo:
    save_fn: Callable  # save function used to save the object to store
    load_fn: Callable  # load function used to load the object from store
    extension: Optional[str] = None  # file extension the object is saved with
    is_binary: Optional[bool] = None  # read, write and append in binary mode
    open_fn: Optional[Callable] = None  # function used to open a file object (default is builtin open())
    needs_path: bool = False  # save and load fn operate on path and not on file like object


class LocalFileStore(ResultsStore):
    def __init__(self, base_dir: str):
        super().__init__()

        self.base_dir = base_dir

        self._type_registry = {
            "event": TypeInfo(self._write, self._read),
            "json": TypeInfo(json.dump, json.load, "json"),
            "pickle": TypeInfo(pickle.dump, pickle.load, "p", is_binary=True),
            "text": TypeInfo(self._write, self._read, "txt"),
        }

        # can be set externally. if set, it is used for naming newly created directories
        self.run_info: Optional["RunInfo"] = None

    @property
    def run_info(self):
        return self._run_info

    @run_info.setter
    def run_info(self, run_info: "RunInfo"):
        self._run_info = run_info

    @staticmethod
    def _write(obj: str, file: IO):
        file.write(obj)

    @staticmethod
    def _read(file: IO) -> str:
        return file.read()

    @staticmethod
    def _get_obj_dir(run_dir: str, sub_dir: Optional[str] = None) -> Optional[str]:
        obj_dir = run_dir
        if sub_dir is not None:
            obj_dir = os.path.join(run_dir, sub_dir)
            os.makedirs(obj_dir, exist_ok=True)
        return obj_dir

    @staticmethod
    def _save_load_info(
        name: str, run_dir: str, obj_dir: str, type_: str, open_kwargs: Dict[str, Any], load_kwargs: Dict[str, Any]
    ):
        load_info = {
            "open_kwargs": open_kwargs,
            "load_kwargs": load_kwargs,
            "obj_dir": os.path.relpath(obj_dir, run_dir),
            "type_": type_,
        }

        # create a hidden load info dir to save object information used by self.load()
        load_info_dir = os.path.join(run_dir, ".load_info")
        os.makedirs(load_info_dir, exist_ok=True)

        pickle.dump(load_info, open(os.path.join(load_info_dir, f'.{name.lstrip(".")}_load_info.p'), "wb"))

    @staticmethod
    def _get_load_info(run_dir: str, name: str) -> Optional[Tuple]:
        # get load information from run dir
        load_info_file_path = os.path.join(run_dir, ".load_info", f'.{name.lstrip(".")}_load_info.p')
        try:
            load_info = pickle.load(open(load_info_file_path, "rb"))
        except FileNotFoundError:
            logger.debug(f'"{name}" could not be found in store, since "{load_info_file_path}" does not exist.')
            return None

        # unpack load info
        type_ = load_info["type_"]
        obj_dir = os.path.join(run_dir, load_info["obj_dir"])
        open_kwargs = load_info["open_kwargs"]
        load_kwargs = load_info["load_kwargs"]
        return type_, obj_dir, open_kwargs, load_kwargs, load_info_file_path

    @staticmethod
    def _create_promise(type_info: TypeInfo, name: str, path: str, open_kwargs: Dict, load_kwargs: Dict) -> FilePromise:
        # load the saved object from run dir
        mode = "rb" if type_info.is_binary else "r"
        if type_info.needs_path:
            mode = None
        return FilePromise(
            name, path, type_info.save_fn, type_info.load_fn, type_info.open_fn, mode, load_kwargs, **open_kwargs
        )

    def save(
        self,
        obj: Any,
        name: str,
        type_: str,
        task_name: str,
        task_unique_config: Dict,
        sub_dir: Optional[str] = None,
        mode: Optional[str] = None,
        open_kwargs: Optional[Dict[str, Any]] = None,
        load_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):

        open_kwargs = {} if open_kwargs is None else open_kwargs
        load_kwargs = {} if load_kwargs is None else load_kwargs

        run_dir = self.get_context(task_name, task_unique_config)

        # get save function and file extension for type
        try:
            type_info = self._type_registry[type_]
        except KeyError:
            raise KeyError(
                f'Object type "{type_}" is not supported in {self.__class__.__name__}. Either extend it by '
                f"implementing specific load and save functions for this type, or save the object as one "
                f'of the following supported types: {", ".join(self._type_registry)}.'
            )

        # set and return save directory for object
        obj_dir = self._get_obj_dir(run_dir, sub_dir)

        # save load info
        self._save_load_info(name, run_dir, obj_dir, type_, open_kwargs, load_kwargs)

        # save object
        name = f"{name}.{type_info.extension}" if type_info.extension else name
        path = os.path.join(obj_dir, name)
        if not mode:
            mode = "wb" if type_info.is_binary else "w"

        if type_info.needs_path:
            type_info.save_fn(obj, path, **kwargs)
        else:
            with File(
                path,
                mode,
                save_fn=type_info.save_fn,
                load_fn=type_info.load_fn,
                open_fn=type_info.open_fn,
                **open_kwargs,
            ) as file:
                file.save(obj=obj, **kwargs)

    def load(self, name: str, task_name: str, task_unique_config: Dict, lazy: bool = False, **kwargs) -> Optional[Any]:
        task_dir = os.path.join(self.base_dir, task_name)

        # try to get existing run dir
        with self.lock:
            run_dir = self._get_run_dir(task_dir=task_dir, task_config=task_unique_config)
        if run_dir is None:
            return None

        # get load information from run dir
        load_info = self._get_load_info(run_dir, name)
        if not load_info:
            return None
        type_, obj_dir, open_kwargs, load_kwargs, load_info_file_path = load_info

        # merge saved load kwargs with user provided kwargs
        #  user provided kwargs overwrite saved kwargs when keys are identical
        load_kwargs = {**load_kwargs, **kwargs}

        # get type info used for file loading
        type_info = self._type_registry[type_]

        # get path
        full_name = f"{name}.{type_info.extension}" if type_info.extension else name
        path = os.path.join(obj_dir, full_name)

        # if path does not exist return None
        if not os.path.exists(path):
            return None

        if lazy:
            return self._create_promise(type_info, name, path, open_kwargs, load_kwargs)

        # load the saved object from run dir
        if type_info.needs_path:
            obj = type_info.load_fn(path, **load_kwargs)
        else:
            mode = "rb" if type_info.is_binary else "r"
            try:
                with File(
                    path,
                    mode,
                    save_fn=type_info.save_fn,
                    load_fn=type_info.load_fn,
                    open_fn=type_info.open_fn,
                    **open_kwargs,
                ) as file:
                    obj = file.load(**load_kwargs)
            except FileNotFoundError:
                logger.warning(
                    f'"{name}" could not be found in store. ' f'Note "{load_info_file_path}" does still exist.'
                )
                return None

        return obj

    def delete(self, name: str, task_name: str, task_unique_config: Dict):
        task_dir = os.path.join(self.base_dir, task_name)

        # try to get existing run dir
        with self.lock:
            run_dir = self._get_run_dir(task_dir=task_dir, task_config=task_unique_config)
        if run_dir is None:
            logger.warning(
                f'"{name}" could not be deleted. '
                f'No run directory for task "{task_name}" and the provided unique_config exists.'
            )
            return None

        # get load information from run dir
        load_info = self._get_load_info(run_dir, name)
        if not load_info:
            return None
        type_, obj_dir, _, _, load_info_file_path = load_info

        # get type info used for file loading
        type_info = self._type_registry[type_]

        name = f"{name}.{type_info.extension}" if type_info.extension else name
        path_to_delete = os.path.join(obj_dir, name)

        # remove the saved object/directory
        try:
            os.remove(path_to_delete)
        except FileNotFoundError:
            logger.warning(f'"{path_to_delete}" could not be deleted from store since it was not found.')
        except IsADirectoryError:
            shutil.rmtree(path_to_delete)

        # remove its load info file from the store
        os.remove(load_info_file_path)

    def delete_run(self, task_name: str, task_unique_config: Dict):
        task_dir = os.path.join(self.base_dir, task_name)

        # try to get existing run dir
        with self.lock:
            run_dir = self._get_run_dir(task_dir=task_dir, task_config=task_unique_config)
        if run_dir is None:
            logger.warning(f'No run directory for task "{task_name}" and the provided unique_config exists.')
            return None

        # delete retrieved run dir
        shutil.rmtree(run_dir)

    def open(
        self,
        name: Optional[str] = None,
        task_name: Optional[str] = None,
        task_unique_config: Optional[Dict] = None,
        mode: Optional[str] = None,
        promise: Optional[FilePromise] = None,
        type_: Optional[str] = None,
        sub_dir: Optional[str] = None,
        **open_kwargs,
    ) -> Optional[File]:

        """
                          | r   r+   w   w+   a   a+   x   x+
        ------------------|-----------------------------------
        read              | +   +        +        +        +
        write             |     +    +   +    +   +    +   +
        write after seek  |     +    +   +             +   +
        create            |          +   +    +   +    +   +
        truncate          |          +   +
        position at start | +   +    +   +             +   +
        position at end   |                   +   +
        """
        assert promise is not None or all(arg is not None for arg in (name, task_name, task_unique_config, mode))

        if promise:
            if mode:
                promise.mode = mode
            return File.from_promise(promise)

        load_kwargs = {}

        if "r" in mode:
            task_dir = os.path.join(self.base_dir, task_name)

            # try to get existing run dir
            with self.lock:
                run_dir = self._get_run_dir(task_dir=task_dir, task_config=task_unique_config)
            if run_dir is None:
                raise FileNotFoundError

            # get load information from run dir
            load_info = self._get_load_info(run_dir, name)
            if not load_info:
                raise FileNotFoundError
            type_, obj_dir, open_kwargs, load_kwargs, load_info_file_path = load_info

            # get type info used for file loading
            type_info = self._type_registry[type_]

        elif "a" in mode:
            # try to get existing run dir
            run_dir = self.get_context(task_name, task_unique_config)

            # get load information from run dir
            load_info = self._get_load_info(run_dir, name)

            if load_info:
                type_, obj_dir, open_kwargs, load_kwargs, load_info_file_path = load_info

                # get type info used for file loading
                type_info = self._type_registry[type_]
            else:
                # get type_info
                try:
                    type_info = self._type_registry[type_]
                except KeyError:
                    raise KeyError(
                        f'Object type "{type_}" is not supported in {self.__class__.__name__}. Either extend it by '
                        f"implementing specific load and save functions for this type, or save the object as one "
                        f'of the following supported types: {", ".join(self._type_registry)}.'
                    )

                # set and return save directory for object
                obj_dir = self._get_obj_dir(run_dir, sub_dir)

                # save load info
                self._save_load_info(name, run_dir, obj_dir, type_, open_kwargs, load_kwargs)

        else:
            run_dir = self.get_context(task_name, task_unique_config)

            # get type_info
            try:
                type_info = self._type_registry[type_]
            except KeyError:
                raise KeyError(
                    f'Object type "{type_}" is not supported in {self.__class__.__name__}. Either extend it by '
                    f"implementing specific load and save functions for this type, or save the object as one "
                    f'of the following supported types: {", ".join(self._type_registry)}.'
                )

            # set and return save directory for object
            obj_dir = self._get_obj_dir(run_dir, sub_dir)

            # save load info
            self._save_load_info(name, run_dir, obj_dir, type_, open_kwargs, load_kwargs)

        # return file object
        name = f"{name}.{type_info.extension}" if type_info.extension else name
        path = os.path.join(obj_dir, name)
        return File(
            path,
            mode,
            load_fn=type_info.load_fn,
            save_fn=type_info.save_fn,
            open_fn=type_info.open_fn,
            load_kwargs=load_kwargs,
            **open_kwargs,
        )

    def get_context(self, task_name: str, task_unique_config: Dict):
        """Method to get the current task's storage context.
        E.g. the current run directory in case of LocalFileStore.
        Creates a new run dir if none exists.
        """
        task_dir = os.path.join(self.base_dir, task_name)

        with self.lock:
            # try to get existing run dir
            run_dir = LocalFileStore._get_run_dir(task_dir=task_dir, task_config=task_unique_config)

            # create new run dir if run dir does not exist
            if run_dir is None:
                run_dir = LocalFileStore._make_run_dir(task_dir=task_dir, run_info=self.run_info)
                json.dump(task_unique_config, open(os.path.join(run_dir, f"config.json"), "w"), indent=4)
        return run_dir

    @staticmethod
    def _scan_task_dir(task_dir: str) -> List[str]:
        os.makedirs(task_dir, exist_ok=True)
        exist_run_dirs = [os.path.join(task_dir, d.name) for d in os.scandir(task_dir) if d.is_dir()]
        return exist_run_dirs

    @staticmethod
    def _get_run_dir(task_dir: str, task_config: Dict) -> Optional[str]:
        exist_run_dirs = LocalFileStore._scan_task_dir(task_dir=task_dir)
        for exist_run_dir in exist_run_dirs:
            try:
                exist_config = json.load(open(os.path.join(exist_run_dir, "config.json"), "r"))
            except FileNotFoundError:
                continue

            if exist_config.items() <= task_config.items():
                return exist_run_dir

        return None

    @staticmethod
    def _make_run_dir(task_dir: str, run_info: Optional["RunInfo"] = None) -> str:
        if run_info is not None:
            run_name = run_info.run_name
            unique_id = run_info.unique_id
            new_dir_name = f"{run_name}__{unique_id}"

        else:
            exist_run_dirs = LocalFileStore._scan_task_dir(task_dir=task_dir)

            dir_names = [os.path.split(d)[-1] for d in exist_run_dirs]

            # if run_info is not None:
            #     run_name = run_info.run_name
            #     # find dirs that start with run_name and extract their suffix (usually a numeric counter)
            #     dir_names = [
            #         name.split(run_name, 1)[-1].replace("-", "") for name in dir_names if name.startswith(run_name)
            #     ]

            # get all numeric dir names in task dir and convert to ids
            ids = [int(d_name) for d_name in dir_names if d_name.isdigit()]

            # increment max id by 1 or start at 0
            new_id = max(ids) + 1 if ids else 0
            new_id = str(new_id).zfill(3)

            new_dir_name = new_id  # if run_info is None else f"{run_info.run_name}-{new_id}"

        # create new run dir
        new_run_dir = os.path.join(task_dir, new_dir_name)
        os.makedirs(new_run_dir, exist_ok=True)
        return new_run_dir
