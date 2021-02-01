from multiprocessing import Manager, Lock
from typing import Dict, Optional, Any

from fluidml.storage import ResultsStore


class InMemoryStore(ResultsStore):
    """ This is an in-memory results store implemented using multiprocessing manager """

    def __init__(self, manager: Manager):
        super().__init__()
        self._memory_store = manager.dict()
        self._lock = Lock()

    def load(self, name: str, task_name: str, task_unique_config: Dict) -> Optional[Any]:
        if task_name not in self._memory_store:
            return None

        for task_sweep in self._memory_store[task_name]:
            if task_sweep["config"] == task_unique_config:
                try:
                    obj = task_sweep['results'][name]
                except KeyError:
                    raise KeyError(f'{name} not saved.')

                return obj

    def save(self, obj: Any, name: str, type_: str, task_name: str, task_unique_config: Dict, **kwargs):
        """ In-memory save function.
        Adds individual object to in-memory store (multiprocessing manager dict).
        """
        with self._lock:
            if task_name not in self._memory_store:
                self._memory_store[task_name] = []

            existing_task_results = self._memory_store[task_name]
            sweep_exists = False
            for task_sweep in existing_task_results:
                if task_sweep['config'] == task_unique_config:
                    task_sweep['results'][name] = obj
                    sweep_exists = True
                    break

            if not sweep_exists:
                new_task_sweep = {'results': {name: obj},
                                  'config': task_unique_config}
                existing_task_results.append(new_task_sweep)

            self._memory_store[task_name] = existing_task_results
