import functools
import inspect
from typing import Union, List, Tuple, Any, TYPE_CHECKING, Dict, Callable, Type

if TYPE_CHECKING:
    from fluidml.flow.task_spec import TaskSpec, Task


class DependencyMixin:
    def __init__(self):
        self._predecessors = []
        self._successors = []

    def requires(self, *predecessors: Union["TaskSpec", List["TaskSpec"]]):
        """Adds predecessor task specs"""
        if len(predecessors) == 1:
            predecessors = predecessors[0] if isinstance(predecessors[0], List) else [predecessors[0]]
        elif any(True if isinstance(task_spec, (List, Tuple)) else False for task_spec in predecessors):
            raise TypeError(
                f"task_spec.requires() either takes a single list of predecessors, "
                f"task_spec.requires([a, b, c]) or a sequence of individual predecessor args"
                f"task_spec.requires(a, b, c)"
            )
        else:
            predecessors = list(predecessors)

        self._predecessors.extend(predecessors)

        # attach this task as a successor to all predecessor tasks
        for predecessor in predecessors:
            predecessor.required_by(self)

    def required_by(self, successor: Any):
        """Adds a successor"""

        self._successors.append(successor)

    @property
    def predecessors(self) -> List["TaskSpec"]:
        return self._predecessors

    @predecessors.setter
    def predecessors(self, predecessors: List["TaskSpec"]):
        self._predecessors = predecessors

    @property
    def successors(self) -> List["TaskSpec"]:
        return self._successors

    @successors.setter
    def successors(self, successors: List["TaskSpec"]):
        self._successors = successors


def publishes(*publish_args, **publish_kwargs):
    """Decorator to register published objects with their respective type annotations.

    Prior to running a task, fluidml checks whether all to-be-published objects exist already.
    If yes, and no "force" execution is configured, the task at hand is skipped and its outputs are forwarded
    to the successor task.

    Example 1:
        ```python
        @publishes(foo=str, bar=Optional[int])
        def run(...):
            pass
        ```
    Object "foo" (type=str) and optional object "bar" (type=int) are published.

    Example 2:
        ```python
        @publishes("foo", "bar")
        def run(...):
            pass
        ```
    Objects "foo" and "bar" are published (no type annotations provided).
    """
    # extract names and type annotations from registered args/kwargs
    published_objects = {name: None for name in publish_args}
    for name, type_annotation in publish_kwargs.items():
        published_objects[name] = type_annotation

    def decorator(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            result = function(*args, **kwargs)
            return result

        wrapper.publishes = published_objects

        return wrapper

    return decorator


def get_expected_args_from_run_signature(
    task: Union[Type["Task"], Callable], config: Dict, additional_kwargs: Dict
) -> Dict:
    if inspect.isclass(task):
        task_all_arguments = dict(inspect.signature(task.run).parameters)
        expected_inputs = {
            arg: value if value.annotation is not inspect.Parameter.empty else None
            for arg, value in task_all_arguments.items()
            if value.kind.name not in ["VAR_POSITIONAL", "VAR_KEYWORD"] and value.name != "self"
        }
    elif inspect.isfunction(task):
        task_all_arguments = dict(inspect.signature(task).parameters)
        task_extra_arguments = list(config) + list(additional_kwargs) + ["task"]
        expected_inputs = {
            arg: value if value.annotation is not inspect.Parameter.empty else None
            for arg, value in task_all_arguments.items()
            if arg not in task_extra_arguments and value.kind.name not in ["VAR_POSITIONAL", "VAR_KEYWORD"]
        }
    else:
        # cannot be reached, check has been made in TaskSpec.
        raise TypeError

    return expected_inputs


def get_published_args_from_run_decorator(task: Union[Type["Task"], Callable]) -> Dict:
    if inspect.isclass(task):
        published_args = task.run.publishes if hasattr(task.run, "publishes") else {}
    elif inspect.isfunction(task):
        published_args = task.publishes if hasattr(task, "publishes") else {}
    else:
        # cannot be reached, check has been made in TaskSpec.
        raise TypeError

    return published_args
