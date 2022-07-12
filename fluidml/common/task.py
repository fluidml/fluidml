import contextlib
import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from multiprocessing import Lock
from typing import Dict, List, Optional, Any, TYPE_CHECKING, Union

from metadict import MetaDict

from fluidml.common.dependency import DependencyMixin
from fluidml.storage import ResultsStore, Promise, File

if TYPE_CHECKING:
    from fluidml.flow import TaskSpec


@dataclass
class Resource(ABC):
    """Dataclass used to register resources, made available to all tasks, e.g. cuda device ids"""


class Task(ABC, DependencyMixin):
    """Base class for a task."""

    def __init__(self):
        DependencyMixin.__init__(self)

        self.project_name: Optional[str] = None
        self.run_name: Optional[str] = None
        self.name: Optional[str] = None
        self.config_kwargs: Optional[Dict[str, Any]] = None
        self.publishes: Optional[List[str]] = None
        self.expects: Optional[Union[List[str], Dict[str, inspect.Parameter]]] = None
        self.id_: Optional[int] = None
        self.unique_config: Optional[MetaDict] = None
        self.reduce: Optional[bool] = None
        self.force: Optional[str] = None
        self.unique_name: Optional[str] = None

        # set in Dolphin or manually
        self._results_store: Optional[ResultsStore] = None
        self._resource: Optional[Resource] = None
        self._lock: Optional[Lock] = None

    @property
    def results_store(self):
        return self._results_store

    @results_store.setter
    def results_store(self, results_store: ResultsStore):
        self._results_store = results_store

    @property
    def resource(self):
        return self._resource

    @resource.setter
    def resource(self, resource: Resource):
        self._resource = resource

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
    def run(self, **results):
        """Implementation of core logic of task.

        Args:
            **results: results from predecessors (automatically passed by flow or swarm (multiprocessing))
        """
        raise NotImplementedError

    def run_wrapped(self, **results):
        """Calls run function to execute task and saves a 'completed' event file to signal successful execution."""
        self.run(**results)
        # todo: if publishes is set, check that all non optional objects are present in saved_objects
        # if self.publishes:
        #     saved_objects: Optional[List] = self.load(name='.saved_objects')
        #     required_objects = [name if not is_optional(annotation) for name, annotation in self.publishes.items()]
        # else:
        #     self.save('1', '.completed', type_='event', sub_dir='.load_info')
        self.save("1", ".completed", type_="event", sub_dir=".load_info")

    def save(self, obj: Any, name: str, type_: Optional[str] = None, **kwargs):
        """Saves the given object to the results store.

        Args:
            obj: Any object that is to be saved
            name: A unique name given to this object
            type_: Additional type specification (eg. json, which is to be passed to results store).
                Defaults to ``None``.
            **kwargs: Additional keyword args.
        """
        with self.lock:
            self.results_store.save(
                obj=obj, name=name, type_=type_, task_name=self.name, task_unique_config=self.unique_config, **kwargs
            )

            # todo: update saved objects file (separate fn)
            # saved_objects: Optional[List] = self.load(
            #     name='.saved_objects', task_name=self.name, task_unique_config=self.unique_config)
            # if saved_objects is None:
            #     saved_objects = []
            # saved_objects.append(name)
            # self.results_store.save(saved_objects, '.saved_objects', type_='json', sub_dir='.load_info',
            #                         task_name=self.name, task_unique_config=self.unique_config)

    def load(
        self,
        name: str,
        task_name: Optional[str] = None,
        task_unique_config: Optional[Union[Dict, MetaDict]] = None,
        **kwargs
    ) -> Any:
        """Loads the given object from results store.

        Args:
            name: An unique name given to this object.
            task_name: Task name which saved the loaded object.
            task_unique_config: Unique config which specifies the run of the loaded object.
            **kwargs: Additional keyword args.
        """
        task_name = task_name if task_name is not None else self.name
        task_unique_config = task_unique_config if task_unique_config is not None else self.unique_config

        return self.results_store.load(name=name, task_name=task_name, task_unique_config=task_unique_config, **kwargs)

    def delete(
        self, name: str, task_name: Optional[str] = None, task_unique_config: Optional[Union[Dict, MetaDict]] = None
    ):
        """Deletes object with specified name from results store"""
        task_name = task_name if task_name is not None else self.name
        task_unique_config = task_unique_config if task_unique_config is not None else self.unique_config

        with self.lock:
            self.results_store.delete(name=name, task_name=task_name, task_unique_config=task_unique_config)

            # todo: update saved objects file (separate fn)
            # saved_objects: Optional[List] = self.load(
            #     name='.saved_objects', task_name=self.name, task_unique_config=self.unique_config)
            # if saved_objects is not None and name in saved_objects:
            #     del saved_objects[saved_objects.index(name)]
            #     self.results_store.save(saved_objects, '.saved_objects', type_='json', sub_dir='.load_info',
            #                             task_name=self.name, task_unique_config=self.unique_config)

    def delete_run(self, task_name: Optional[str] = None, task_unique_config: Optional[Union[Dict, MetaDict]] = None):
        """Deletes run with specified name from results store"""
        task_name = task_name if task_name is not None else self.name
        task_unique_config = task_unique_config if task_unique_config is not None else self.unique_config

        with self.lock:
            self.results_store.delete_run(task_name=task_name, task_unique_config=task_unique_config)

    def get_store_context(
        self, task_name: Optional[str] = None, task_unique_config: Optional[Union[Dict, MetaDict]] = None
    ) -> Any:
        """Wrapper to get store specific storage context, e.g. the current run directory for Local File Store"""
        task_name = task_name if task_name is not None else self.name
        task_unique_config = task_unique_config if task_unique_config is not None else self.unique_config

        with self.lock:
            return self.results_store.get_context(task_name=task_name, task_unique_config=task_unique_config)

    def open(
        self,
        name: Optional[str] = None,
        task_name: Optional[str] = None,
        task_unique_config: Optional[Union[Dict, MetaDict]] = None,
        mode: Optional[str] = None,
        promise: Optional[Promise] = None,
        type_: Optional[str] = None,
        sub_dir: Optional[str] = None,
        **open_kwargs
    ) -> Optional[File]:
        """Wrapper to open a file from Local File Store (only available for Local File Store)."""

        if promise:
            with self.lock:
                return self.results_store.open(promise=promise, mode=mode)

        task_name = task_name if task_name is not None else self.name
        task_unique_config = task_unique_config if task_unique_config is not None else self.unique_config

        with self.lock:
            return self.results_store.open(
                name=name,
                task_name=task_name,
                task_unique_config=task_unique_config,
                mode=mode,
                type_=type_,
                sub_dir=sub_dir,
                **open_kwargs
            )

    @classmethod
    def from_spec(cls, task_spec: "TaskSpec"):
        # avoid circular import
        from fluidml.common.utils import MyTask

        # convert task config values to MetaDicts
        task_spec.config_kwargs = MetaDict(task_spec.config_kwargs)
        task_spec.additional_kwargs = MetaDict(task_spec.additional_kwargs)

        if inspect.isclass(task_spec.task):
            task = task_spec.task(**task_spec.config_kwargs, **task_spec.additional_kwargs)
            task.config_kwargs = task_spec.config_kwargs
            task_all_arguments = dict(inspect.signature(task.run).parameters)
            expected_inputs = {
                arg: value
                for arg, value in task_all_arguments.items()
                if value.kind.name not in ["VAR_POSITIONAL", "VAR_KEYWORD"]
            }
        elif inspect.isfunction(task_spec.task):
            task = MyTask(
                task=task_spec.task,
                config_kwargs=task_spec.config_kwargs,
                additional_kwargs=task_spec.additional_kwargs,
            )

            task_all_arguments = dict(inspect.signature(task_spec.task).parameters)
            task_extra_arguments = list(task_spec.config_kwargs) + list(task_spec.additional_kwargs) + ["task"]
            expected_inputs = {
                arg: value
                for arg, value in task_all_arguments.items()
                if arg not in task_extra_arguments and value.kind.name not in ["VAR_POSITIONAL", "VAR_KEYWORD"]
            }
        else:
            # cannot be reached, check has been made in TaskSpec.
            raise TypeError

        task.project_name = task_spec.project_name
        task.run_name = task_spec.run_name
        task.results_store = task_spec.results_store
        task.resource = task_spec.resource
        task.name = task_spec.name
        task.unique_name = task_spec.unique_name
        task.id_ = task_spec.id_
        task.unique_config = task_spec.unique_config
        task.reduce = task_spec.reduce
        task.force = task_spec.force
        task.predecessors = task_spec.predecessors
        task.successors = task_spec.successors

        # set task publishes attribute based on user provided task spec or task values
        #  set for both task and task_spec
        task = Task._set_task_publishes(task_spec, task)

        # set task expects attribute based on user provided task spec or task value and the run method signature
        #  set for both task and task_spec
        task = Task._set_task_expects(task_spec, task, expected_inputs)

        return task

    @staticmethod
    def _set_task_expects(task_spec: "TaskSpec", task: "Task", expected_inputs: Dict[str, inspect.Parameter]) -> "Task":
        # if expects is provided to task_spec manually we add missing arguments to the expected_inputs dict
        if task_spec.expects is not None:
            for arg in task_spec.expects:
                if arg not in expected_inputs:
                    expected_inputs[arg] = inspect.Parameter(name=arg, kind=inspect.Parameter.POSITIONAL_OR_KEYWORD)

        # if expects is provided to task manually we add missing arguments to the expected_inputs dict
        if task.expects is not None:
            for arg in task.expects:
                if arg not in expected_inputs:
                    expected_inputs[arg] = inspect.Parameter(name=arg, kind=inspect.Parameter.POSITIONAL_OR_KEYWORD)

        task.expects = expected_inputs
        task_spec.expects = expected_inputs
        return task

    @staticmethod
    def _set_task_publishes(task_spec: "TaskSpec", task: "Task") -> "Task":
        if task_spec.publishes is not None:
            task.publishes = task_spec.publishes
        else:
            task.publishes = []
            task_spec.publishes = []
        return task
