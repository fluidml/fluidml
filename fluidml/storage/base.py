import contextlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from multiprocessing import RLock
from typing import Any, Dict, Optional

from metadict import MetaDict

from fluidml.utils import change_logging_level

logger = logging.getLogger(__name__)


class Names(str, Enum):
    FLUIDML_INFO_FILE = "fluidml_info"
    SAVED_RESULTS_FILE = ".saved_results"
    FLUIDML_DIR = ".fluidml"
    CONFIG = "config.json"

    def __str__(self):
        return self.value


class Promise(ABC):
    """An interface for future objects, that can be lazy loaded."""

    @abstractmethod
    def load(self, **kwargs):
        """Loads the actual object."""
        raise NotImplementedError


@dataclass
class Sweep:
    """A sweep class holding the value and config of a specific task result.

    List of Sweeps are the standard inputs for ``reduce`` tasks that gather results from grid search expanded tasks.
    """

    value: Any
    """The value of the object."""
    config: MetaDict
    """The unique config of the object#s task."""


@dataclass
class LazySweep:
    """A lazy variation of the ``Sweep`` class."""

    value: Promise
    """The lazy value of the object."""
    config: MetaDict
    """The unique config of the object#s task."""


@dataclass
class StoreContext:
    """The store context of the current task."""

    run_dir: Optional[str] = None
    """The run directory of the task. Relevant for File Stores."""
    sweep_counter: Optional[str] = None
    """The sweep counter of the task. Relevant for File Stores. A dynamically created counter to distinguish different 
    task instances with the same run name in the results store."""


class ResultsStore(ABC):
    """The interface of a results store."""

    def __init__(self):
        self._lock: Optional[RLock] = None

    @property
    def lock(self):
        # if no lock was provided we return a dummy contextmanager that does nothing
        if self._lock is None:
            null_context = contextlib.suppress()
            return null_context
        return self._lock

    @lock.setter
    def lock(self, lock: RLock):
        self._lock = lock

    @abstractmethod
    def load(self, name: str, task_name: str, task_unique_config: Dict, **kwargs) -> Optional[Any]:
        """Loads the given object from results store based on its name, task_name and task_config if it exists.

        Args:
            name: A unique name given to this object.
            task_name: Task name which saved the loaded object.
            task_unique_config: Unique config which specifies the run of the loaded object.
            **kwargs: Additional keyword argumentss.

        Returns:
            The loaded object.
        """
        raise NotImplementedError

    @abstractmethod
    def save(
        self,
        obj: Any,
        name: str,
        type_: str,
        task_name: str,
        task_unique_config: Dict,
        **kwargs,
    ):
        """Method to save/update any artifact.

        Args:
            obj: The object to save/update
            name: The object name.
            type\_: The type of the object. Only required for file stores.
            task_name: The task name that saves/updates the object.
            task_unique_config: The unique config of that task.
            **kwargs: Additional keyword arguments.
        """
        raise NotImplementedError

    @abstractmethod
    def delete(self, name: str, task_name: str, task_unique_config: Dict):
        """Method to delete any artifact.

        Args:
            name: The object name.
            task_name: The task name that saved the object.
            task_unique_config: Unique config which specifies the run of the object.
        """
        raise NotImplementedError

    @abstractmethod
    def delete_run(self, task_name: str, task_unique_config: Dict):
        """Method to delete all task results from a given run config

        Args:
            task_name: Task name which saved the object.
            task_unique_config: Unique config which specifies the run of the object.
        """
        raise NotImplementedError

    def open(
        self,
        name: Optional[str] = None,
        task_name: Optional[str] = None,
        task_unique_config: Optional[Dict] = None,
        mode: Optional[str] = None,
        promise: Optional[Promise] = None,
        type_: Optional[str] = None,
        sub_dir: Optional[str] = None,
        **open_kwargs,
    ):
        """Method to open a file from Local File Store (only available for file stores).

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
            A ``File`` object that behaves like a file like object.
        """

    @abstractmethod
    def get_context(self, task_name: str, task_unique_config: Dict) -> StoreContext:
        """Wrapper to get store specific storage context, e.g. the current run directory for Local File Store

        Args:
            task_name: Task name.
            task_unique_config: Unique config which specifies the run.

        Returns:
            The store context object holding information like the current run dir.
        """
        raise NotImplementedError

    def get_results(self, task_name: str, task_unique_config: Dict) -> Optional[Dict]:
        """Collects all saved results based that have been tracked when using ``Task.save()``.

        Args:
            task_name: Task name which saved the object.
            task_unique_config: Unique config which specifies the run of the object.

        Returns:
            A dictionary of all saved result objects.
        """

        # try to load saved results
        saved_results = self.load(
            Names.SAVED_RESULTS_FILE,
            task_name=task_name,
            task_unique_config=task_unique_config,
        )

        # if a no saved results can be found, we return None and exit
        if not saved_results:
            return None

        # here we loop over individual item names and call user provided self.load() to get individual item data
        results = {}
        for item_name in saved_results:
            # load object
            obj: Optional[Any] = self.load(
                name=item_name,
                task_name=task_name,
                task_unique_config=task_unique_config,
            )

            if obj is not None:
                # store object in results
                results[item_name] = obj

        return results

    def is_finished(self, task_name: str, task_unique_config: Dict) -> bool:
        """Checks if a task is finished.

        Args:
            task_name: Task name which saved the object.
            task_unique_config: Unique config which specifies the run of the object.

        Returns:
            A boolean indicating whether the task is finished or not.
        """
        from fluidml.task import TaskState

        # try to load task completed object; if it is None we return None and re-run the task
        with change_logging_level(40):
            run_info: Optional[Dict] = self.load(
                name=Names.FLUIDML_INFO_FILE,
                task_name=task_name,
                task_unique_config=task_unique_config,
            )
        if run_info and run_info["state"] == TaskState.FINISHED:
            return True
        else:
            return False
