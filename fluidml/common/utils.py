from typing import Any, Dict, Callable, Optional

from fluidml.common import Task, Resource


class MyTask(Task):
    """A constructor class that creates a task object from a callable."""

    def __init__(self,
                 name: str,
                 id_: int,
                 task: Callable,
                 kwargs: Dict,
                 force: Optional[bool] = None):
        super().__init__(id_=id_, name=name)
        self.task = task
        self.kwargs = kwargs
        self.force = force

    def run(self, results: Dict[str, Any], resource: Resource):
        result = self.task(results, resource, **self.kwargs)
        return result
