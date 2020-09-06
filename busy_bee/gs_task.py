from busy_bee.task import Task
from typing import Dict, Any


class GridSearcheableTask:
    def __init__(self, task: Task, gs_config: Dict[str, Any]):
        """
        A wrapper class for making tasks to be grid searcheable

        Args:
            task (Task): [description]
            gs_config (Dict[str, Any]): [description]
        """
        self._task = task
        self._gs_config = gs_config

    # method to make interfaces of task available
    # https://stackoverflow.com/questions/1466676/create-a-wrapper-class-to-call-a-pre-and-post-function-around-existing-functions/1467296
    def __getattr__(self, attr):
        orig_attr = self._task.__getattribute__(attr)
        if callable(orig_attr):
            def hooked(*args, **kwargs):
                result = orig_attr(*args, **kwargs)
                return result
            return hooked
        else:
            return orig_attr

    @property
    def wrapped_task(self):
        return self._task

    @property
    def gs_config(self):
        return self._gs_config