import json
import logging
import os
import pickle
import shutil
from dataclasses import dataclass
from typing import IO, TYPE_CHECKING, Any, AnyStr, Callable, Dict, List, Optional, Tuple

from fluidml.storage.base import Names, Promise, ResultsStore, StoreContext

if TYPE_CHECKING:
    from fluidml.task import TaskInfo

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
    """A file like wrapper class to support opening files using the ``LocalFileStore``.

    Args:
        path: The path to the file to open.
        mode: The open mode, e.g. "r", "w", etc.
        save_fn: A callable used for ``file.save()`` calls.
        load_fn: A callable used for ``file.load()`` calls.
        open_fn: An optional callable used for file opening. The default is the inbuilt ``open()`` function.
        load_kwargs: Additional keyword arguments passed to the open function.
    """

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
        """Creates a ``File`` object from a `´FilePromise``."""
        return cls(
            promise.path,
            promise.mode,
            promise.save_fn,
            promise.load_fn,
            promise.open_fn,
            **promise.open_kwargs,
        )


@dataclass
class TypeInfo:
    """Initializes saving and loading information for a specific type."""

    save_fn: Callable
    """Save function used to save the object to store."""
    load_fn: Callable
    """Load function used to load the object from store."""
    extension: Optional[str] = None
    """File extension the object is saved with."""
    is_binary: Optional[bool] = None
    """Read, write and append in binary mode."""
    open_fn: Optional[Callable] = None
    """Function used to open a file object (default is builtin open())."""
    needs_path: bool = False
    """Whether save and load fn should operate on path and not on file like object. Default: false."""


class LocalFileStore(ResultsStore):
    """A local file store that allows to easily save and load task results to/from a base directory in a file system.

    Out of the box the local file store supports three common file types, "json", "pickle" and "text".
    It can be easily extended to arbitrary file types by subclassing the LocalFileStore and registering new
    Types to the `self._type_registry` dictionary. A new type needs to register a load and save function
    using the ``TypeInfo`` data class.

    Args:
        base_dir: The base directory that is used to store results from tasks.

    Attributes:
        base_dir: The base directory that is used to store results from tasks.
        type_registry: The dictionary to register new types with a save and load function.
    """

    def __init__(self, base_dir: str):
        super().__init__()

        self.base_dir = base_dir

        self.type_registry: Dict[str, TypeInfo] = {
            "event": TypeInfo(self._write, self._read),
            "json": TypeInfo(json.dump, json.load, "json"),
            "pickle": TypeInfo(pickle.dump, pickle.load, "p", is_binary=True),
            "text": TypeInfo(self._write, self._read, "txt"),
        }

        # can be set externally. if set, it is used for naming newly created directories
        self._run_info: Optional["TaskInfo"] = None

    @property
    def run_info(self):
        """The current run info of a task, which is used for naming newly created directories."""
        return self._run_info

    @run_info.setter
    def run_info(self, run_info: "TaskInfo"):
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
        name: str,
        run_dir: str,
        obj_dir: str,
        type_: str,
        open_kwargs: Dict[str, Any],
        load_kwargs: Dict[str, Any],
    ):
        load_info = {
            "open_kwargs": open_kwargs,
            "load_kwargs": load_kwargs,
            "obj_dir": os.path.relpath(obj_dir, run_dir),
            "type_": type_,
        }

        # create a hidden load info dir to save object information used by self.load()
        load_info_dir = os.path.join(run_dir, Names.FLUIDML_DIR)
        os.makedirs(load_info_dir, exist_ok=True)

        pickle.dump(
            load_info,
            open(os.path.join(load_info_dir, f'.{name.lstrip(".")}_load_info.p'), "wb"),
        )

    @staticmethod
    def _get_load_info(run_dir: str, name: str) -> Optional[Tuple]:
        # get load information from run dir
        load_info_file_path = os.path.join(run_dir, Names.FLUIDML_DIR, f'.{name.lstrip(".")}_load_info.p')

        try:
            load_info = pickle.load(open(load_info_file_path, "rb"))
        except FileNotFoundError:
            load_info = None

        # Fallback to old load info dir name ".load_info" -> for backward compatibility with old results
        if load_info is None:
            load_info_old_file_path = os.path.join(run_dir, ".load_info", f'.{name.lstrip(".")}_load_info.p')
            try:
                load_info = pickle.load(open(load_info_old_file_path, "rb"))
            except FileNotFoundError:
                logger.warning(f'"{name}" could not be found in store, since "{load_info_file_path}" does not exist.')
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
            name,
            path,
            type_info.save_fn,
            type_info.load_fn,
            type_info.open_fn,
            mode,
            load_kwargs,
            **open_kwargs,
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
        """Saves an object to the local file store.

        If no task and run directory for the provided unique config exists, a new directory will be created.

        Args:
            obj: The object to save.
            name: An unique name given to this object.
            type\_: Additional type specification (e.g. json, which is to be passed to results store).
            task_name: Task name which saves the object.
            task_unique_config: Unique config which specifies the run of the object.
            sub_dir: A path of a subdirectory used for saving the file.
            mode: The mode to save the file, e.g. "w" or "wb".
            open_kwargs: Additional keyword arguments passed to the registered ``open()`` function.
            load_kwargs: Additional keyword arguments passed to the registered ``load()`` function.
            **kwargs: Additional keyword arguments passed to the registered ``save()`` function.
        """

        open_kwargs = {} if open_kwargs is None else open_kwargs
        load_kwargs = {} if load_kwargs is None else load_kwargs

        run_dir = self.get_context(task_name, task_unique_config).run_dir

        # get save function and file extension for type
        try:
            type_info = self.type_registry[type_]
        except KeyError:
            raise KeyError(
                f'Object type "{type_}" is not supported in {self.__class__.__name__}. Either extend it by '
                f"implementing specific load and save functions for this type, or save the object as one "
                f'of the following supported types: {", ".join(self.type_registry)}.'
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

    def load(
        self,
        name: str,
        task_name: str,
        task_unique_config: Dict,
        lazy: bool = False,
        **kwargs,
    ) -> Optional[Any]:
        """Loads an object from the local file store.

        The object is retrieved based on the name and the provided task name and unique task config.

        Args:
            name: An unique name given to this object.
            task_name: Task name which saves the object.
            task_unique_config: Unique config which specifies the run of the object.
            lazy: A boolean whether the object should be lazily loaded. If True, a ``FilePromise`` object will be
                returned, that can be loaded into memory on demand with the ``promise.load()`` function.
            **kwargs: Additional keyword arguments passed to the registered ``load()`` function.

        Returns:
            The specified object if it is found.
        """
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
        type_info = self.type_registry[type_]

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
                logger.warning(f'Task "{task_name}" could not find "{name}" in results store. ')
                return None

        return obj

    def delete(self, name: str, task_name: str, task_unique_config: Dict):
        """Deletes an object from the local file store.

        The object is deleted based on the name and the provided task name and unique task config.

        Args:
            name: The name of the to be deleted object.
            task_name: Task name which saved the object.
            task_unique_config: Unique config which specifies the run of the object.
        """

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
        type_info = self.type_registry[type_]

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
        """Deletes an entire task run directory from the local file store.

        The run is deleted based on the task name and the unique task config.

        Args:
            task_name: The name of the task.
            task_unique_config: Unique config which specifies the run of the object.
        """
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
        """Wrapper to open a file from Local File Store (only available for Local File Store).

        It returns a file like object that has additional ``save()`` and ``load()`` methods that can be used to
        save/load objects with a registered type to/from the file store.
        The ``File`` like object allows for an incremental write or read process of objects that for example don't fit
        into memory.

        Args:
            name: An unique name given to this object.
            task_name: Task name which saved the object.
            task_unique_config: Unique config which specifies the run of the object.
            mode: The open mode, e.g. "r", "w", etc.
            promise: An optional ``Promise`` object used for creating the returned file like object.
            type\_: Additional type specification (e.g. json, which is to be passed to results store).
            sub_dir: A path of a subdirectory used for opening the file.
            **open_kwargs: Additional keyword arguments passed to the registered ``open()`` function.

        Returns:
            A ``File`` object that wraps a file like object and enables incremental result store reading and writing.
        """

        # The available file pointer methods per open mode are:
        #                       | r   r+   w   w+   a   a+   x   x+
        #     ------------------|-----------------------------------
        #     read              | +   +        +        +        +
        #     write             |     +    +   +    +   +    +   +
        #     write after seek  |     +    +   +             +   +
        #     create            |          +   +    +   +    +   +
        #     truncate          |          +   +
        #     position at start | +   +    +   +             +   +
        #     position at end   |                   +   +

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
            type_info = self.type_registry[type_]

        elif "a" in mode:
            # try to get existing run dir
            run_dir = self.get_context(task_name, task_unique_config).run_dir

            # get load information from run dir
            load_info = self._get_load_info(run_dir, name)

            if load_info:
                (
                    type_,
                    obj_dir,
                    open_kwargs,
                    load_kwargs,
                    load_info_file_path,
                ) = load_info

                # get type info used for file loading
                type_info = self.type_registry[type_]
            else:
                # get type_info
                try:
                    type_info = self.type_registry[type_]
                except KeyError:
                    raise KeyError(
                        f'Object type "{type_}" is not supported in {self.__class__.__name__}. Either extend it by '
                        f"implementing specific load and save functions for this type, or save the object as one "
                        f'of the following supported types: {", ".join(self.type_registry)}.'
                    )

                # set and return save directory for object
                obj_dir = self._get_obj_dir(run_dir, sub_dir)

                # save load info
                self._save_load_info(name, run_dir, obj_dir, type_, open_kwargs, load_kwargs)

        else:
            run_dir = self.get_context(task_name, task_unique_config).run_dir

            # get type_info
            try:
                type_info = self.type_registry[type_]
            except KeyError:
                raise KeyError(
                    f'Object type "{type_}" is not supported in {self.__class__.__name__}. Either extend it by '
                    f"implementing specific load and save functions for this type, or save the object as one "
                    f'of the following supported types: {", ".join(self.type_registry)}.'
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

    def get_context(self, task_name: str, task_unique_config: Dict) -> StoreContext:
        """Method to get the current task's storage context.

        E.g. the current run directory in case of LocalFileStore.
        Creates a new run dir if none exists.

        Args:
            task_name: Task name.
            task_unique_config: Unique config which specifies the run.
        """
        task_dir = os.path.join(self.base_dir, task_name)

        with self.lock:
            # try to get existing run dir
            run_dir = LocalFileStore._get_run_dir(task_dir=task_dir, task_config=task_unique_config)

            # create new run dir if run dir does not exist
            if run_dir is None:
                run_dir = LocalFileStore._make_run_dir(task_dir=task_dir, run_info=self.run_info)
                json.dump(
                    task_unique_config,
                    open(os.path.join(run_dir, Names.CONFIG), "w"),
                    indent=4,
                )

            sweep_counter = os.path.split(run_dir)[-1].rsplit("-")[-1]
        return StoreContext(run_dir=run_dir, sweep_counter=sweep_counter)

    @staticmethod
    def _scan_task_dir(task_dir: str) -> List[str]:
        if not os.path.isdir(task_dir):
            return []
        exist_run_dirs = [os.path.join(task_dir, d.name) for d in os.scandir(task_dir) if d.is_dir()]
        return exist_run_dirs

    @staticmethod
    def _get_run_dir(task_dir: str, task_config: Dict) -> Optional[str]:
        exist_run_dirs = LocalFileStore._scan_task_dir(task_dir=task_dir)
        for exist_run_dir in exist_run_dirs:
            try:
                exist_config = json.load(open(os.path.join(exist_run_dir, Names.CONFIG), "r"))
            except FileNotFoundError:
                continue

            if exist_config.items() <= task_config.items():
                return exist_run_dir

        return None

    @staticmethod
    def _make_run_dir(task_dir: str, run_info: Optional["TaskInfo"] = None) -> str:
        # if run info exists and holds sweep_counter attribute, we create the previously existent run dir name
        if run_info and run_info.sweep_counter:
            new_dir_name = f"{run_info.run_name}-{run_info.sweep_counter}"
        else:
            # retrieve all existing dir names for task
            exist_run_dirs = LocalFileStore._scan_task_dir(task_dir=task_dir)
            dir_names = [os.path.split(d)[-1] for d in exist_run_dirs]

            # if run info exists we use the assigned run_name to filter the relevant run dir names
            if run_info:
                run_name = run_info.run_name
                # find dirs that start with run_name and extract their suffix (usually a numeric counter)
                dir_names = [name.rsplit("-", 1)[-1] for name in dir_names if name.startswith(run_name)]

            # get all numeric dir names in task dir and convert to ids
            ids = [int(d_name) for d_name in dir_names if d_name.isdigit()]

            # increment max id by 1 or start at 0
            new_id = max(ids) + 1 if ids else 0
            new_id = str(new_id).zfill(3)

            # create the new run dir name
            new_dir_name = new_id if run_info is None else f"{run_info.run_name}-{new_id}"

        # create and return new run dir
        new_run_dir = os.path.join(task_dir, new_dir_name)
        os.makedirs(new_run_dir, exist_ok=True)
        return new_run_dir
