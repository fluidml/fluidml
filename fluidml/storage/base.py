import contextlib
import functools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from multiprocessing import Lock
from typing import Optional, Dict, Any

from metadict import MetaDict

from fluidml.common.utils import is_optional


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


class InheritDecoratorsMixin:
    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        decorator_registry = getattr(cls, "_decorator_registry", {}).copy()
        cls._decorator_registry = decorator_registry
        # Check for decorated objects in the mixin itself (optional):
        for name, obj in __class__.__dict__.items():
            if getattr(obj, "inherit_decorator", False) and name not in decorator_registry:
                decorator_registry[name] = obj.inherit_decorator
        # annotate newly decorated methods in the current class/subclass:
        for name, obj in cls.__dict__.items():
            if getattr(obj, "inherit_decorator", False) and name not in decorator_registry:
                decorator_registry[name] = obj.inherit_decorator
        # finally, decorate all methods annotated in the registry:
        for name, decorator in decorator_registry.items():
            if name in cls.__dict__ and getattr(getattr(cls, name), "inherit_decorator", None) != decorator:
                setattr(cls, name, decorator(cls.__dict__[name]))


# TODO (LH): Add decorators to save and delete fn
def _save_object_names(func):
    """Decorator to save names of saved objects.

    The resulting ".saved_objects" name list is used to check if any or all to-be-published objects of a task
    have been saved before.
    """

    @functools.wraps(func)
    def wrapper(self, obj: Any, name: str, type_: str, task_name: str, task_unique_config: Dict, **kwargs):
        # save actual object
        res = func(self, obj, name, type_, task_name, task_unique_config, **kwargs)

        # log name of saved object
        saved_objects: Optional[Dict[str, None]] = self.load(
            name=".saved_objects", task_name=task_name, task_unique_config=task_unique_config
        )
        if saved_objects is None:
            saved_objects = {}
        if name not in saved_objects:
            saved_objects[name] = None
        func(
            self,
            obj=saved_objects,
            name=".saved_objects",
            type_="json",
            sub_dir=".load_info",
            task_name=task_name,
            task_unique_config=task_unique_config,
        )
        return res

    # necessary for the decorator inheritance to work
    wrapper.inherit_decorator = _save_object_names
    return wrapper


def _delete_object_names(func):
    """Decorator to delete names of previously saved but now deleted objects."""

    @functools.wraps(func)
    def wrapper(self, obj: Any, name: str, type_: str, task_name: str, task_unique_config: Dict, **kwargs):
        # delete actual object
        res = func(self, obj, name, type_, task_name, task_unique_config, **kwargs)

        # load saved object names
        saved_objects: Optional[Dict[str, None]] = self.load(
            name=".saved_objects", task_name=task_name, task_unique_config=task_unique_config
        )
        # delete name from registry and save registry
        if saved_objects is not None and name in saved_objects:
            del saved_objects[name]
            func(
                self,
                obj=saved_objects,
                name=".saved_objects",
                type_="json",
                sub_dir=".load_info",
                task_name=task_name,
                task_unique_config=task_unique_config,
            )
        return res

    # necessary for the decorator inheritance to work
    wrapper.inherit_decorator = _delete_object_names
    return wrapper


class ResultsStore(ABC, InheritDecoratorsMixin):
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
        **open_kwargs
    ):
        """Method to open a file from Local File Store (only available for Local File Store)."""

    def get_context(self, task_name: str, task_unique_config: Dict):
        """Method to get store specific storage context, e.g. the current run directory for Local File Store"""

    def get_results(self, task_name: str, task_unique_config: Dict, task_publishes: Dict[str, Any]) -> Optional[Dict]:
        # if a task publishes no results, we always execute the task
        if not task_publishes:
            return None

        # if a task is not yet finished, we again execute the task
        if not self.is_finished(task_name=task_name, task_unique_config=task_unique_config):
            return None

        # here we loop over individual item names and call user provided self.load() to get individual item data
        results = {}
        for item_name, type_annotation in task_publishes.items():
            # load object
            obj: Optional[Any] = self.load(name=item_name, task_name=task_name, task_unique_config=task_unique_config)

            # if at least one expected and non-optional result object of the task cannot be loaded,
            # return None and re-run the task.
            if not is_optional(type_annotation) and obj is None:
                return None
            # if obj is None:
            #     return None

            # store object in results
            results[item_name] = obj

        return results

    # TODO(LH): determine best functionality to decide whether a task can be skipped
    def is_finished(self, task_name: str, task_unique_config: Dict) -> bool:
        # try to load task completed object; if it is None we return None and re-run the task
        completed: Optional[str] = self.load(
            name=".completed", task_name=task_name, task_unique_config=task_unique_config
        )
        if not completed:
            return False
        return True
