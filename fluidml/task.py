import datetime
import inspect
import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

from metadict import MetaDict

from fluidml.dependency import DependencyMixin
from fluidml.storage.base import Names, Promise, ResultsStore, StoreContext
from fluidml.storage.file_store import File
from fluidml.utils import BaseModel, change_logging_level

if TYPE_CHECKING:
    from fluidml.task_spec import TaskSpec

logger = logging.getLogger(__name__)


class TaskResults(BaseModel):
    """Holds the results and unique config of a task."""

    task_name: str
    """The name of the task, e.g. "training"."""
    task_unique_name: str
    """The unique name of the task, e.g. "training-3", in the context of the current run."""
    unique_config: Dict
    """The unique config specifying the task."""
    results: Optional[Dict[str, Any]] = None
    """The results dictionary. Each key-value pair represents a result name and value."""


class TaskState(str, Enum):
    """The state of a task."""

    PENDING = "pending"
    """Task has not been scheduled and processed, yet."""
    SCHEDULED = "scheduled"
    """Task has been scheduled for processing."""
    RUNNING = "running"
    """Task is currently running."""
    KILLED = "killed"
    """Task has been killed by the user via KeyBoardInterrupt."""
    FAILED = "failed"
    """Task failed due to an unexpected error."""
    UPSTREAM_FAILED = "upstream_failed"
    """Task failed/could not be executed due to one or more upstream task failures."""
    FINISHED = "finished"
    """Task finished successfully."""


class TaskInfo(BaseModel):
    """A wrapper class to store important task information.

    Used for serializing the information to the results store.
    """

    project_name: str
    """Name of the project."""
    run_name: str
    """Name of the task's current run."""
    state: Optional[TaskState] = None
    """The current state of the task, e.g. "running", "finished", etc."""
    started: Optional[datetime.datetime] = None
    """The start time of the task."""
    ended: Optional[datetime.datetime] = None
    """The end time of the task."""
    duration: Optional[datetime.timedelta] = None
    """The current running time of the task."""
    run_history: Optional[Dict] = None
    """Holds the task ids of all predecessor task including the task itself."""
    sweep_counter: Optional[str] = None
    """A dynamically created counter to distinguish different task instances with the same run name 
    in the results store. E.g. used in the LocalFileStore to name run directories.
    """
    unique_config_hash: Optional[str] = None
    """An 8 character hash of the unique run config."""
    id: Optional[str] = None
    """A combination of the run-name and the sweep counter or unique config hash if the former does not exist."""


class Task(ABC, DependencyMixin):
    """The base ``Task`` that provides the user with additional features such as result store and run name access.

    When implementing a task as a Class

    Args:
        name: Name of the task, e.g. "Processing".
        unique_name: Unique name of the task if a run contains multiple instances of the same task,
            e.g. "Processing-3".
        project_name: Name of the project.
        run_name: Name of the task's current run.
        state: The current state of the task, e.g. "running", "finished", etc.
        started: The start time of the task.
        ended: The end time of the task.
        run_history: Holds the task ids of all predecessor task including the task itself.
        sweep_counter: A dynamically created counter to distinguish different task instances with the same run name
            in the results store. E.g. used in the LocalFileStore to name run directories.
        unique_config_hash: An 8 character hash of the unique run config.
        unique_config: Unique config of the task. Includes all predecessor task configs as well to uniquely define a
            task.
        results_store: An instance of results store for results management. If nothing is provided, a non-persistent
            InMemoryStore store is used.
        resource: A resource object that can hold arbitrary data, e.g. gpu or cpu device information.
            Resource objects can be assigned in a multiprocessing context, so that each worker process
            uses a dedicated resource, e.g. cuda device.
        force: Indicator if the task is force executed or not.
        expects: A dict of expected input arguments and their :func:`inspect.Parameter` objects
            of the task's run method signature.
        reduce: A boolean indicating whether this is a reduce-task. Defaults to ``None``.
        config: The config dictionary of the task. Only contains the config parameters of the task (no predecessors).
    """

    # needed to avoid overwriting already initialized Task attributes, when the user task's __init__ method
    #  with a super() call is called at a later time.
    _is_initialized: bool = False

    def __init__(
        self,
        name: Optional[str] = None,
        unique_name: Optional[str] = None,
        project_name: Optional[str] = None,
        run_name: Optional[str] = None,
        state: Optional[TaskState] = None,
        started: Optional[datetime.datetime] = None,
        ended: Optional[datetime.datetime] = None,
        run_history: Optional[Dict] = None,
        sweep_counter: Optional[str] = None,
        unique_config_hash: Optional[str] = None,
        unique_config: Optional[Union[Dict, MetaDict]] = None,
        results_store: Optional[ResultsStore] = None,
        resource: Optional[Any] = None,
        force: Optional[bool] = None,
        expects: Optional[Dict[str, inspect.Parameter]] = None,
        reduce: Optional[bool] = None,
        config: Optional[Dict[str, Any]] = None,
    ):

        if not self._is_initialized:
            DependencyMixin.__init__(self)

            self.name = name
            self.unique_name = unique_name
            self.project_name = project_name
            self.run_name = run_name
            self.state = state
            self.started = started
            self.ended = ended
            self.run_history = run_history
            self.sweep_counter = sweep_counter
            self.unique_config_hash = unique_config_hash
            self.unique_config = unique_config
            self.results_store = results_store
            self.resource = resource
            self.force = force
            self.expects = expects
            self.reduce = reduce
            self.config = config

            self._is_initialized = True

    @property
    def id(self) -> str:
        """A combination of the run-name and the sweep counter or unique config hash if the former does not exist."""
        return (
            f"{self.run_name}-{self.sweep_counter}"
            if self.sweep_counter
            else f"{self.run_name}-{self.unique_config_hash}"
        )

    @property
    def duration(self) -> Optional[datetime.timedelta]:
        """The current running time of the task."""
        if self.started and self.ended:
            return self.ended - self.started
        elif self.started:
            return datetime.datetime.now() - self.started
        else:
            return None

    @property
    def info(self) -> TaskInfo:
        """A ``TaskInfo`` object that summarizes the important task information for serialization."""
        return TaskInfo(
            project_name=self.project_name,
            run_name=self.run_name,
            state=self.state,
            started=self.started,
            ended=self.ended,
            duration=self.duration,
            run_history=self.run_history,
            sweep_counter=self.sweep_counter,
            unique_config_hash=self.unique_config_hash,
            id=self.id,
        )

    @abstractmethod
    def run(self, **results):
        """Implementation of core logic of task.

        Args:
            **results: results from predecessors (automatically passed)
        """
        raise NotImplementedError

    def _init(self):
        """A wrapper to call the actual user task's __init__ method with the provided config"""
        self.__init__(**self.config)

    @staticmethod
    def _check_no_internally_used_config_keys(task: "Task", config: MetaDict):
        internal_keys = [k for k in config if k in {**Task.__dict__, **task.__dict__}]
        if internal_keys:
            raise ValueError(
                f"The config keys: {', '.join(internal_keys)} are protected since they are used "
                f"internally by the Task class. Please consider renaming them."
            )

    def _track_saved_object(self, name: str, mode: str, type_: Optional[str] = None):
        # load saved objects
        with change_logging_level(40):
            saved_objects: Optional[List[str]] = self.load(
                name=Names.SAVED_RESULTS_FILE,
                task_name=self.name,
                task_unique_config=self.unique_config,
            )

        if mode == "save":
            if saved_objects is None:
                saved_objects = []

            if name not in saved_objects:
                saved_objects.append(name)
                self.results_store.save(
                    obj=saved_objects,
                    name=Names.SAVED_RESULTS_FILE,
                    type_="json",
                    sub_dir=Names.FLUIDML_DIR,
                    task_name=self.name,
                    task_unique_config=self.unique_config,
                )

        elif mode == "delete":
            # delete name from registry and save registry
            if saved_objects is not None and name in saved_objects:
                del saved_objects[saved_objects.index(name)]
                self.results_store.save(
                    obj=saved_objects,
                    name=Names.SAVED_RESULTS_FILE,
                    type_="json",
                    sub_dir=Names.FLUIDML_DIR,
                    task_name=self.name,
                    task_unique_config=self.unique_config,
                )
        else:
            raise ValueError(f'"mode" argument is "{mode}" but must be "save" or "delete".')

        debug_msg = f'Task "{self.unique_name}" {mode}s "{name}"'
        msg = debug_msg + f"." if type_ is None else debug_msg + f' of type "{type_}".'
        logger.debug(msg)

    def save(self, obj: Any, name: str, type_: Optional[str] = None, **kwargs):
        """Saves the given object to the results store.

        Args:
            obj: Any object that is to be saved
            name: A unique name given to this object
            type\_: Additional type specification (e.g. json, which is to be passed to results store).
                Defaults to ``None``.
            **kwargs: Additional keyword args.
        """
        self.results_store.save(
            obj=obj,
            name=name,
            type_=type_,
            task_name=self.name,
            task_unique_config=self.unique_config,
            **kwargs,
        )
        self._track_saved_object(name, mode="save", type_=type_)

    def load(
        self,
        name: str,
        task_name: Optional[str] = None,
        task_unique_config: Optional[Union[Dict, MetaDict]] = None,
        **kwargs,
    ) -> Any:
        """Loads the given object from results store.

        Args:
            name: A unique name given to this object.
            task_name: Task name which saved the loaded object.
            task_unique_config: Unique config which specifies the run of the loaded object.
            **kwargs: Additional keyword args.

        Returns:
            The loaded object.
        """
        task_name = task_name if task_name is not None else self.name
        task_unique_config = task_unique_config if task_unique_config is not None else self.unique_config

        return self.results_store.load(
            name=name,
            task_name=task_name,
            task_unique_config=task_unique_config,
            **kwargs,
        )

    def delete(
        self,
        name: str,
        task_name: Optional[str] = None,
        task_unique_config: Optional[Union[Dict, MetaDict]] = None,
    ):
        """Deletes object with specified name from results store

        Args:
            name: A unique name given to this object.
            task_name: Task name which saved the object.
            task_unique_config: Unique config which specifies the run of the object.
        """
        task_name = task_name if task_name is not None else self.name
        task_unique_config = task_unique_config if task_unique_config is not None else self.unique_config

        self.results_store.delete(name=name, task_name=task_name, task_unique_config=task_unique_config)
        self._track_saved_object(name, mode="delete")

    def delete_run(
        self,
        task_name: Optional[str] = None,
        task_unique_config: Optional[Union[Dict, MetaDict]] = None,
    ):
        """Deletes run with specified name from results store

        Args:
            task_name: Task name which saved the object.
            task_unique_config: Unique config which specifies the run of the object.
        """
        task_name = task_name if task_name is not None else self.name
        task_unique_config = task_unique_config if task_unique_config is not None else self.unique_config

        self.results_store.delete_run(task_name=task_name, task_unique_config=task_unique_config)

    def get_store_context(
        self,
        task_name: Optional[str] = None,
        task_unique_config: Optional[Union[Dict, MetaDict]] = None,
    ) -> StoreContext:
        """Wrapper to get store specific storage context, e.g. the current run directory for Local File Store

        Args:
            task_name: Task name.
            task_unique_config: Unique config which specifies the run.

        Returns:
            The store context object holding information like the current run dir.
        """
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
        """Wrapper to open a file from Local File Store (only available for Local File Store).

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

        if promise:
            return self.results_store.open(promise=promise, mode=mode)

        task_name = task_name if task_name is not None else self.name
        task_unique_config = task_unique_config if task_unique_config is not None else self.unique_config

        if mode is not None and ("w" in mode or "a" in mode):
            self._track_saved_object(name, mode="save", type_=type_)

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
    def from_spec(cls, task_spec: "TaskSpec", half_initialize: bool = False) -> "Task":
        """Initializes a Task object from a TaskSpec object.

        Args:
            task_spec: A task specification object.
            half_initialize: A boolean to indicate whether only the parent Task object is initialized and the
                child class initialization is delayed until ``task._init()`` is called.
                The half initialization is only needed internally to create a task object without directly executing the
                ``__init__`` method of the actual task implemented by the user.

        Returns:
            A task object (fully or half initialized).
        """

        # convert task config values to MetaDict
        task_spec.config = MetaDict(task_spec.config)

        if inspect.isclass(task_spec.task):
            if half_initialize:
                # create a new user task object without initialization
                task = task_spec.task.__new__(task_spec.task)
                # only init the inherited base task
                Task.__init__(task)
            else:
                # normal initialization
                task = task_spec.task(**task_spec.config)
            task.config = task_spec.config

            # make sure the user provided config does not contain first level keys that are used in the Task class
            Task._check_no_internally_used_config_keys(task=task, config=task.config)

        elif inspect.isfunction(task_spec.task):
            # create an artificial wrapper task object to support functional tasks
            task = _TaskFromCallable(
                task=task_spec.task,
                config=task_spec.config,
            )
        else:
            # cannot be reached, check has been made in TaskSpec.
            raise TypeError

        task.name = task_spec.name
        task.unique_name = task_spec.unique_name
        task.unique_config = task_spec.unique_config
        task.reduce = task_spec.reduce
        task.expects = task_spec.expects
        task.predecessors = task_spec.predecessors
        task.successors = task_spec.successors

        return task


class _TaskFromCallable(Task):
    """A wrapper class that wraps a callable as a Task."""

    def __init__(self, task: Callable, config: MetaDict):
        super().__init__()
        self.task = task
        self.config = config

    def run(self, **results: Dict[str, Any]):
        self.task(**results, **self.config, task=self)

    def _init(self):
        pass
