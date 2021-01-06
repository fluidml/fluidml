from copy import deepcopy
from multiprocessing import Queue, Lock
from queue import Empty
from typing import Dict, Any, List, Optional, Tuple, Union

from fluidml.common import Task, Resource
from fluidml.common.logging import Console
from fluidml.swarm import Whale
from fluidml.storage import ResultsStore
from fluidml.storage.utils import pack_predecessor_results
from fluidml.common.utils import MyTask


class Dolphin(Whale):
    def __init__(self,
                 id_: int,
                 resource: Resource,
                 scheduled_queue: Queue,
                 running_queue: List[int],
                 done_queue: List[int],
                 lock: Lock,
                 tasks: Dict[int, Task],
                 exception: Dict[str, Exception],
                 exit_on_error: bool,
                 results_store: Optional[ResultsStore] = None):
        super().__init__(exception=exception, exit_on_error=exit_on_error)
        self.id_ = id_
        self.resource = resource
        self.scheduled_queue = scheduled_queue
        self.running_queue = running_queue
        self.done_queue = done_queue
        self.lock = lock
        self.tasks = tasks
        self.results_store = results_store

    def _is_task_ready(self, task: Task):
        for predecessor in task.predecessors:
            if predecessor.id_ not in self.done_queue:
                return False
        return True

    def _extract_results_from_predecessors(self, task: Task) -> Dict[str, Any]:
        predecessor_tasks_with_store = []
        for predecessor in task.predecessors:
            predecessor.results_store = self.results_store
            predecessor.results_store.task_name = predecessor.name
            predecessor.results_store.task_unique_config = predecessor.unique_config
            predecessor.results_store.task_publishes = predecessor.publishes
            predecessor_tasks_with_store.append(deepcopy(predecessor))

        results: Dict = pack_predecessor_results(predecessor_tasks_with_store, task.reduce)
        return results

    def _run_task(self, task: Task, pred_results: Dict):
        with self.lock:
            # try to get results from results store
            results: Optional[Tuple[Dict, str]] = task.results_store.get_results()
        # if results is none or force is set, run the task now
        if results is None or task.force:
            Console.get_instance().log(
                f'Dolphin {self.id_} started running task {task.name}-{task.id_}.')
            if isinstance(task, MyTask):
                task.run(results=pred_results)
            else:
                task.run(**pred_results)
            Console.get_instance().log(
                f'Dolphin {self.id_} completed running task {task.name}-{task.id_}.')

        # take results from results store and continue
        else:
            Console.get_instance().log(
                f'Task {task.name}-{task.id_} already executed.')

    def _pack_task(self, task: Task) -> Task:
        task.results_store = self.results_store
        task.results_store.task_name = task.name
        task.results_store.task_unique_config = task.unique_config
        task.results_store.task_publishes = task.publishes
        task.resource = self.resource
        return task

    def _execute_task(self, task: Task):
        # extract predecessor results
        with self.lock:
            pred_results = self._extract_results_from_predecessors(task)
            self.running_queue.append(task.id_)

        # pack the task
        task = self._pack_task(task)

        # run the task
        self._run_task(task, pred_results)

        with self.lock:
            # put task in done_queue
            self.done_queue.append(task.id_)

    def _fetch_next_task(self) -> Union[int, None]:
        task_id = None
        try:
            task_id = self.scheduled_queue.get(block=False, timeout=0.5)
        except Empty:
            pass
            # Console.get_instance().log(f'Dolphin {self.id_}: waiting for tasks.')
        return task_id

    def _done(self) -> bool:
        return self.exception or (len(self.done_queue) == len(self.tasks))

    def _work(self):
        while not self._done():
            # fetch next task to run
            task_id = self._fetch_next_task()

            # continue when there is a valid task to run
            if task_id is not None:

                # get current task from task_id
                task = self.tasks[task_id]

                # TODO: Do we need a lock here? -> Yes, without I saw how 2 workers executed same task.
                with self.lock:
                    # run task only if it has not been executed already
                    if task_id in self.done_queue or task_id in self.running_queue:
                        Console.get_instance().log(
                            f'Task {task.name}-{task_id} is currently running or already finished.')
                        continue

                # all good to execute the task
                self._execute_task(task)

                # schedule the task's successors
                self._schedule_successors(task)

    def _schedule_successors(self, task: Task):
        # get successor tasks and put them in task queue for processing
        for successor in task.successors:
            # run task only if all dependencies are satisfied
            if not self._is_task_ready(task=self.tasks[successor.id_]):
                Console.get_instance().log(
                    f'Dolphin {self.id_}: Dependencies are not satisfied yet for '
                    f'task {successor.name}-{successor.id_}')
            elif successor.id_ in self.done_queue or successor.id_ in self.running_queue:
                Console.get_instance().log(f'Task {successor.name}-{successor.id_} '
                                           f'is currently running or already finished.')
            else:
                Console.get_instance().log(
                    f'Dolphin {self.id_} is now scheduling {successor.name}-{successor.id_}.')
                self.scheduled_queue.put(successor.id_)
