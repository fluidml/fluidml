from typing import Union, List, Any, TYPE_CHECKING


if TYPE_CHECKING:
    from fluidml.common import Task
    from fluidml.flow.task_spec import BaseTaskSpec


class DependencyMixin:
    def __init__(self):
        self._predecessors = []
        self._successors = []

    def requires(self, predecessors: Union['BaseTaskSpec', List['BaseTaskSpec'], List['Task']]):
        """Adds predecessor task specs"""
        predecessors = predecessors if isinstance(predecessors, List) else [predecessors]

        self._predecessors.extend(predecessors)

        # attach this task as a successor to all predecessor tasks
        for predecessor in predecessors:
            predecessor.required_by(self)

    def required_by(self, successor: Any):
        """Adds a successor"""

        self._successors.append(successor)

    @property
    def predecessors(self) -> Union[List['Task'], List['BaseTaskSpec']]:
        return self._predecessors

    @property
    def successors(self) -> Union[List['Task'], List['BaseTaskSpec']]:
        return self._successors
