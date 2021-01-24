import logging
import logging.handlers
import multiprocessing
from multiprocessing import Manager, set_start_method, Queue, Lock
import random
from types import TracebackType
from typing import Optional, Type, List, Dict, Union, Any

from rich.logging import RichHandler

from fluidml.common.logging import FluidLogger
from fluidml.common import Task, Resource
from fluidml.swarm import Dolphin, Orca
from fluidml.storage import ResultsStore, InMemoryStore
from fluidml.storage.utils import pack_results


class Swarm:
    def __init__(self,
                 n_dolphins: Optional[int] = None,
                 resources: Optional[List[Resource]] = None,
                 results_store: Optional[ResultsStore] = None,
                 start_method: str = 'spawn',
                 refresh_every: Optional[int] = None,
                 exit_on_error: bool = True,
                 return_results: bool = False,
                 logging_handler: Optional[RichHandler] = None):
        set_start_method(start_method, force=True)
        self.logging_handler = logging_handler if logging_handler is not None else RichHandler(rich_tracebacks=True,
                                                                                               markup=True)
        self._configure_logging()

        self.n_dolphins = n_dolphins if n_dolphins else multiprocessing.cpu_count()
        self.resources = Swarm._allocate_resources(self.n_dolphins, resources)
        self.manager = Manager()
        self.lock = Lock()

        self.scheduled_queue = Queue()
        self.running_queue = self.manager.list()
        self.done_queue = self.manager.list()
        self.logging_queue = Queue()

        self.results_store = results_store if results_store is not None else InMemoryStore(
            self.manager, self.lock)
        self.exception = self.manager.dict()
        self.return_results = True if isinstance(self.results_store, InMemoryStore) else return_results
        self.tasks: Dict[int, Task] = {}

        self.fluid_logger = FluidLogger(logging_queue=self.logging_queue)

        # orca worker for tracking
        self.dolphins = [Orca(done_queue=self.done_queue,
                              logging_queue=self.logging_queue,
                              tasks=self.tasks,
                              exception=self.exception,
                              exit_on_error=exit_on_error,
                              refresh_every=refresh_every)]
        # dolphin workers for task execution
        self.dolphins.extend([Dolphin(id_=i,
                                      resource=self.resources[i],
                                      scheduled_queue=self.scheduled_queue,
                                      running_queue=self.running_queue,
                                      done_queue=self.done_queue,
                                      logging_queue=self.logging_queue,
                                      lock=self.lock,
                                      tasks=self.tasks,
                                      exception=self.exception,
                                      exit_on_error=exit_on_error,
                                      results_store=self.results_store) for i in range(self.n_dolphins)])

    def __enter__(self):
        return self

    def __exit__(self,
                 exc_type: Optional[Type[BaseException]],
                 exc_val: Optional[BaseException],
                 exc_tb: Optional[TracebackType]):
        self.close()

    def _configure_logging(self):
        root = logging.getLogger()
        formatter = logging.Formatter('%(processName)-10s\n%(message)s')
        self.logging_handler.setFormatter(formatter)
        root.addHandler(self.logging_handler)
        root.setLevel(logging.DEBUG)

    @staticmethod
    def _allocate_resources(n_dolphins: int, resources: List[Resource]) -> List[Resource]:
        if not resources:
            resources = [None] * n_dolphins
        elif len(resources) != n_dolphins:
            # we assign resources to dolphins uniformly (from uniform distribution)
            resources = random.choices(resources, k=n_dolphins)
        return resources

    @staticmethod
    def _get_entry_point_tasks(tasks: List[Task]) -> Dict[int, str]:
        """
        Gets tasks that are run first (tasks with no predecessors)
        """
        entry_task_ids = {}
        for task in tasks:
            if len(task.predecessors) == 0:
                entry_task_ids[task.id_] = task.name
        return entry_task_ids

    def _collect_results(self) -> Dict[str, Any]:
        results = pack_results(all_tasks=list(self.tasks.values()),
                               results_store=self.results_store,
                               return_results=self.return_results)
        return results

    def work(self, tasks: List[Task]) -> Optional[Dict[str, Union[List[Dict], Dict]]]:
        logger = logging.getLogger(__name__)

        # get entry point task ids
        entry_point_tasks: Dict[int, str] = self._get_entry_point_tasks(tasks)

        # also update the current tasks
        for task in tasks:
            self.tasks[task.id_] = task

        # start the fluid logger
        self.fluid_logger.start()

        # schedule entry point tasks
        for task_id, task_name in entry_point_tasks.items():
            # Console.get_instance().log(
            #     f'Swarm scheduling task {task_name}-{task_id}.')
            logger.info(f'Swarm scheduling task {task_name}-{task_id}.')
            self.scheduled_queue.put(task_id)

        # start the workers
        for dolphin in self.dolphins:
            dolphin.start()

        # wait for them to finish
        for dolphin in self.dolphins:
            dolphin.join()

        # join the fluid logger
        self.logging_queue.put(None)
        self.fluid_logger.join()

        # if an exception was raised by a child process, re-raise it again in the parent.
        if self.exception:
            raise self.exception['message']

        # return results
        return self._collect_results()

    def close(self):
        self.tasks = self.manager.dict()
        self.done_queue = self.manager.dict()
        for dolphin in self.dolphins:
            dolphin.close()
