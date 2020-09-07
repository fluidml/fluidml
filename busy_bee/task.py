from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass(init=True)
class Resource:
    pass


class Task(ABC):
    """ Abstract class for task
    """
    def __init__(self, id_: int, name: str):
        self.id_ = id_
        self.name = name
        self.predecessors = []
        self.successors = []

    def requires(self, tasks: List['Task']):
        """
        Adds predecessor tasks that need to be run before this task

        Args:
            tasks (List[): tasks
        """
        self.predecessors.extend(tasks)

        # attach this task as a successor to all predecessor tasks
        for task in tasks:
            task._required_by(self)

    def _required_by(self, task: 'Task'):
        """
        Adds a successor task that needs to be scheduled after this task

        Args:
            task (Task): [description]
        """
        self.successors.append(task)

    @abstractmethod
    def run(self, results: Dict[str, Any], resource: Resource) -> Optional[Dict[str, Any]]:
        """
        Implementation of core logic of task

        Args:
            results (Dict[str, Any]): results from predecessors (automatically passed by swarm)
            resource (Resource): resource to use (automatically passed by swarm)

        Returns:
            Optional[Dict[str, Any]]: a dict of results (automatically passed to successor tasks)
        """
        raise NotImplementedError
