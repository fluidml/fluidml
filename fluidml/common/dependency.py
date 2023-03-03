from typing import Union, List, Tuple, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from fluidml.common.task import Task
    from fluidml.flow.task_spec import TaskSpec


class DependencyMixin:
    """Mixin to register dependencies between tasks.

    Properties:
        predecessors: A List of predecessor `TaskSpec` objects, registered to this task.
        successors: A List of successor `TaskSpec` objects, registered to this task.
    """

    def __init__(self):
        self._predecessors = []
        self._successors = []

    def requires(self, *predecessors: Union["TaskSpec", List["TaskSpec"], "Task", List["Task"]]):
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
    def predecessors(self) -> List[Union["TaskSpec", "Task"]]:
        return self._predecessors

    @predecessors.setter
    def predecessors(self, predecessors: List[Union["TaskSpec", "Task"]]):
        self._predecessors = predecessors

    @property
    def successors(self) -> List[Union["TaskSpec", "Task"]]:
        return self._successors

    @successors.setter
    def successors(self, successors: List[Union["TaskSpec", "Task"]]):
        self._successors = successors
