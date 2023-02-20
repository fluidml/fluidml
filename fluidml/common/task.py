import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, TYPE_CHECKING, Union, Callable

from metadict import MetaDict

from fluidml.common.dependency import DependencyMixin
from fluidml.storage import ResultsStore, Promise, File

if TYPE_CHECKING:
    from fluidml.flow import TaskSpec


# TODO (LH): Change to Pydantic in the future for easier json serialization
@dataclass
class RunInfo:
    project_name: str
    run_name: str
    unique_id: Optional[str] = None
    run_path: Dict = field(default_factory=dict)

    def dict(self) -> Dict:
        return self.__dict__


class Task(ABC, DependencyMixin):
    """Base class for a task."""

    def __init__(self):
        DependencyMixin.__init__(self)

        self.run_info: Optional[RunInfo] = None
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
        self._resource: Optional[Any] = None

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
    def resource(self, resource: Any):
        self._resource = resource

    @abstractmethod
    def run(self, **results):
        """Implementation of core logic of task.

        Args:
            **results: results from predecessors (automatically passed by flow or swarm (multiprocessing))
        """
        raise NotImplementedError

    def run_wrapped(self, **results):
        """Calls run function to execute task and saves a 'completed' event file to signal successful execution."""
        self.save(self.run_info.dict(), "fluidml_run_info", type_="json", indent=4)
        self.run(**results)
        # TODO (LH): if publishes is set, check that all non optional objects are present in saved_objects
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
            type_: Additional type specification (e.g. json, which is to be passed to results store).
                Defaults to ``None``.
            **kwargs: Additional keyword args.
        """
        self.results_store.save(
            obj=obj, name=name, type_=type_, task_name=self.name, task_unique_config=self.unique_config, **kwargs
        )

    def load(
        self,
        name: str,
        task_name: Optional[str] = None,
        task_unique_config: Optional[Union[Dict, MetaDict]] = None,
        **kwargs,
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

        self.results_store.delete(name=name, task_name=task_name, task_unique_config=task_unique_config)

    def delete_run(self, task_name: Optional[str] = None, task_unique_config: Optional[Union[Dict, MetaDict]] = None):
        """Deletes run with specified name from results store"""
        task_name = task_name if task_name is not None else self.name
        task_unique_config = task_unique_config if task_unique_config is not None else self.unique_config

        self.results_store.delete_run(task_name=task_name, task_unique_config=task_unique_config)

    def get_store_context(
        self, task_name: Optional[str] = None, task_unique_config: Optional[Union[Dict, MetaDict]] = None
    ) -> Any:
        """Wrapper to get store specific storage context, e.g. the current run directory for Local File Store"""
        task_name = task_name if task_name is not None else self.name
        task_unique_config = task_unique_config if task_unique_config is not None else self.unique_config

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
        **open_kwargs,
    ) -> Optional[File]:
        """Wrapper to open a file from Local File Store (only available for Local File Store)."""

        if promise:
            return self.results_store.open(promise=promise, mode=mode)

        task_name = task_name if task_name is not None else self.name
        task_unique_config = task_unique_config if task_unique_config is not None else self.unique_config

        return self.results_store.open(
            name=name,
            task_name=task_name,
            task_unique_config=task_unique_config,
            mode=mode,
            type_=type_,
            sub_dir=sub_dir,
            **open_kwargs,
        )

    @classmethod
    def from_spec(cls, task_spec: "TaskSpec"):

        # convert task config values to MetaDicts
        task_spec.config = MetaDict(task_spec.config)
        task_spec.additional_kwargs = MetaDict(task_spec.additional_kwargs)

        # TODO (LH): Deep merge config kwargs and additional kwargs (to preserve original config structure of arguments)

        if inspect.isclass(task_spec.task):
            task = task_spec.task(**task_spec.config, **task_spec.additional_kwargs)
            task.config = task_spec.config
        elif inspect.isfunction(task_spec.task):
            task = _TaskFromCallable(
                task=task_spec.task,
                config_kwargs=task_spec.config,
                additional_kwargs=task_spec.additional_kwargs,
            )
        else:
            # cannot be reached, check has been made in TaskSpec.
            raise TypeError

        task.run_info = task_spec.run_info
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
        task.expects = task_spec.expects
        task.publishes = task_spec.publishes

        return task


class _TaskFromCallable(Task):
    """A wrapper class that wraps a callable as a Task."""

    def __init__(self, task: Callable, config_kwargs: MetaDict, additional_kwargs: MetaDict):
        super().__init__()
        self.task = task
        self.config_kwargs = config_kwargs
        self.additional_kwargs = additional_kwargs

    def run(self, **results: Dict[str, Any]):
        self.task(**results, **self.config_kwargs, **self.additional_kwargs, task=self)
