import logging
from multiprocessing import Manager
from typing import Any, Dict, Optional

from fluidml.storage.base import ResultsStore, StoreContext

logger = logging.getLogger(__name__)


class InMemoryStore(ResultsStore):
    """An in-memory results store implemented using a dict.

    When multiprocessing is used a ``manager.dict()`` is used for inter process communication.
    If no result store is provided to ``flow.run()`` this in-memory store is the default.
    """

    def __init__(self, manager: Optional[Manager] = None):
        super().__init__()
        self._memory_store = manager.dict() if manager is not None else {}

    def load(self, name: str, task_name: str, task_unique_config: Dict, **kwargs) -> Optional[Any]:
        with self.lock:
            if task_name not in self._memory_store:
                return None

            for task_sweep in self._memory_store[task_name]:
                if task_sweep["config"].items() <= task_unique_config.items():
                    try:
                        obj = task_sweep["results"][name]
                    except KeyError:
                        logger.warning(f'Task "{task_name}" could not find "{name}" in result store.')
                        return None

                    return obj

    def save(
        self,
        obj: Any,
        name: str,
        task_name: str,
        task_unique_config: Dict,
        type_: Optional[str] = None,
        **kwargs,
    ):
        """In-memory save function.
        Adds individual object to in-memory store.
        """
        with self.lock:
            if task_name not in self._memory_store:
                self._memory_store[task_name] = []

            existing_task_results = self._memory_store[task_name]
            sweep_exists = False
            for task_sweep in existing_task_results:
                if task_sweep["config"] == task_unique_config:
                    task_sweep["results"][name] = obj
                    sweep_exists = True
                    break

            if not sweep_exists:
                new_task_sweep = {"results": {name: obj}, "config": task_unique_config}
                existing_task_results.append(new_task_sweep)

            self._memory_store[task_name] = existing_task_results

    def delete(self, name: str, task_name: str, task_unique_config: Dict):
        with self.lock:
            if task_name not in self._memory_store:
                logger.warning(f'"{name}" could not be deleted. ' f"Task {task_name} does not exist in InMemoryStore.")
                return None

            existing_task_results = self._memory_store[task_name]

            for task_sweep in existing_task_results:
                if task_sweep["config"].items() <= task_unique_config.items():
                    try:
                        del task_sweep["results"][name]
                        self._memory_store[task_name] = existing_task_results
                    except KeyError:
                        logger.warning(f'"{name}" could not be deleted from store since it was not found.')
                    return None

            logger.warning(
                f'"{name}" could not be deleted. ' f'No matching unique_config for task "{task_name}" exists.'
            )

    def delete_run(self, task_name: str, task_unique_config: Dict):
        with self.lock:
            if task_name not in self._memory_store:
                logger.warning(f"Task {task_name} does not exist in InMemoryStore.")
                return None

            existing_task_results = self._memory_store[task_name]

            self._memory_store[task_name] = [
                task_sweep
                for task_sweep in existing_task_results
                if not task_sweep["config"].items() <= task_unique_config.items()
            ]

    def get_context(self, task_name: str, task_unique_config: Dict) -> StoreContext:
        pass
