from typing import Union, List, Tuple, Any, TYPE_CHECKING


if TYPE_CHECKING:
    from fluidml.flow.task_spec import BaseTaskSpec, TaskSpec


class DependencyMixin:
    def __init__(self):
        self._predecessors = []
        self._successors = []

    def requires(self, *predecessors: Union['BaseTaskSpec', List['BaseTaskSpec'], List['TaskSpec']]):
        """Adds predecessor task specs"""
        if len(predecessors) == 1:
            predecessors = predecessors[0] if isinstance(predecessors[0], List) else [predecessors[0]]
        elif any(True if isinstance(task_spec, (List, Tuple)) else False for task_spec in predecessors):
            raise TypeError(f'task_spec.requires() either takes a single list of predecessors, '
                            f'task_spec.requires([a, b, c]) or a sequence of individual predecessor args'
                            f'task_spec.requires(a, b, c)')
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
    def predecessors(self) -> Union[List['TaskSpec'], List['BaseTaskSpec']]:
        return self._predecessors

    @predecessors.setter
    def predecessors(self, predecessors: Union[List['TaskSpec'], List['BaseTaskSpec']]):
        self._predecessors = predecessors

    @property
    def successors(self) -> Union[List['TaskSpec'], List['BaseTaskSpec']]:
        return self._successors

    @successors.setter
    def successors(self, successors: Union[List['TaskSpec'], List['BaseTaskSpec']]):
        self._successors = successors
