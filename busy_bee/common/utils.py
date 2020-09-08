from typing import Any, Dict, Callable

from busy_bee.common import Task, Resource


class MyTask(Task):
    def __init__(self,
                 id_: int,
                 name: str,
                 task: Callable,
                 kwargs: Dict):
        super().__init__(id_=id_, name=name)
        self.task = task
        self.kwargs = kwargs

    def run(self, results: Dict[str, Any], resource: Resource):
        result = self.task(**self.kwargs)
        return result
