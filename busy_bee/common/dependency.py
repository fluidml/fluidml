from typing import List, Any


class DependencyMixin(object):
    def __init__(self):
        self._predecessors = []
        self._successors = []

    def requires(self, predecessors: List[Any]):
        """
        Adds predecessor task specs
        """
        self._predecessors.extend(predecessors)

        # attach this task as a successor to all predecessor tasks
        for predecessor in predecessors:
            predecessor._required_by(self)

    def _required_by(self, successor: Any):
        """
        Adds a successor
        """
        self._successors.append(successor)

    @property
    def predecessors(self) -> List[Any]:
        return self._predecessors

    @property
    def successors(self) -> List[Any]:
        return self._successors
