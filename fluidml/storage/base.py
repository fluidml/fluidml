import contextlib

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from multiprocessing import Lock
from typing import Optional, Dict, Any, List

from metadict import MetaDict


logger = logging.getLogger(__name__)


class Promise(ABC):
    @abstractmethod
    def load(self, **kwargs):
        raise NotImplementedError


@dataclass
class Sweep:
    value: Any
    config: MetaDict


@dataclass
class LazySweep:
    value: Promise
    config: MetaDict


@dataclass
class StoreContext:
    run_dir: Optional[str] = None
    sweep_counter: Optional[str] = None


class ResultsStore(ABC):
    def __init__(self):
        self._lock: Optional[Lock] = None

    @property
    def lock(self):
        # if no lock was provided we return a dummy contextmanager that does nothing
        if self._lock is None:
            null_context = contextlib.suppress()
            return null_context
        return self._lock

    @lock.setter
    def lock(self, lock: Lock):
        self._lock = lock

    @abstractmethod
    def load(self, name: str, task_name: str, task_unique_config: Dict, **kwargs) -> Optional[Any]:
        """Query method to load an object based on its name, task_name and task_config if it exists"""
        raise NotImplementedError

    @abstractmethod
    # @_save_object_names
    def save(self, obj: Any, name: str, type_: str, task_name: str, task_unique_config: Dict, **kwargs):
        """Method to save/update any artifact"""
        raise NotImplementedError

    @abstractmethod
    # @_delete_object_names
    def delete(self, name: str, task_name: str, task_unique_config: Dict):
        """Method to delete any artifact"""
        raise NotImplementedError

    @abstractmethod
    def delete_run(self, task_name: str, task_unique_config: Dict):
        """Method to delete all task results from a given run config"""
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
        """Method to open a file from Local File Store (only available for Local File Store)."""

    @abstractmethod
    def get_context(self, task_name: str, task_unique_config: Dict) -> StoreContext:
        """Method to get store specific storage context, e.g. the current run directory for Local File Store"""

    def get_results(self, task_name: str, task_unique_config: Dict, saved_results: List[str]) -> Optional[Dict]:
        # if a task publishes no results, we always execute the task
        if not saved_results:
            return None

        # if a task is not yet finished, we again execute the task
        if not self.is_finished(task_name=task_name, task_unique_config=task_unique_config):
            return None

        # here we loop over individual item names and call user provided self.load() to get individual item data
        results = {}
        for item_name in saved_results:
            # load object
            obj: Optional[Any] = self.load(name=item_name, task_name=task_name, task_unique_config=task_unique_config)

            # if at least one expected and non-optional result object of the task cannot be loaded,
            # return None and re-run the task.
            if obj is None:
                return None

            # store object in results
            results[item_name] = obj

        return results

    def is_finished(self, task_name: str, task_unique_config: Dict) -> bool:
        from fluidml.common.task import TaskState

        # try to load task completed object; if it is None we return None and re-run the task
        run_info: Optional[Dict] = self.load(
            name="fluidml_run_info", task_name=task_name, task_unique_config=task_unique_config
        )
        if not run_info:
            return False
        elif run_info["state"] == TaskState.FINISHED:
            return True
        else:
            return True

    # def is_finished(self, task_name: str, task_unique_config: Dict) -> bool:
    #     # try to load task completed object; if it is None we return None and re-run the task
    #     completed: Optional[str] = self.load(
    #         name=".completed", task_name=task_name, task_unique_config=task_unique_config
    #     )
    #     if not completed:
    #         return False
    #     return True
