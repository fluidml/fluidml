from multiprocessing import Queue, Lock
from queue import Empty
from typing import Dict, Any, List, Optional, Tuple, Union

from fluidml.common.task import Task, Resource
from fluidml.common.logging import Console
from fluidml.swarm import Whale
from fluidml.storage import ResultsStore
from fluidml.storage.utils import pack_results


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
        task_configs = [(predecessor.name, predecessor.unique_config) for predecessor in task.predecessors]
        results = pack_results(self.results_store, task_configs)
        return results

    def _run_task(self, task: Task, pred_results: Dict):
        with self.lock:
            # try to get results from results store
            results: Optional[Tuple[Dict, str]] = self.results_store.get_results(task_name=task.name,
                                                                                 unique_config=task.unique_config)
        # if results is none or force is set, run the task now
        if results is None or task.force:
            Console.get_instance().log(f'Dolphin {self.id_} started running task {task.name}-{task.id_}.')
            results: Dict = task.run(results=pred_results, resource=self.resource)
            Console.get_instance().log(f'Dolphin {self.id_} completed running task {task.name}-{task.id_}.')

            # save/update the results
            with self.lock:
                save_function = self.results_store.update_results if task.force else self.results_store.save_results
                save_function(task_name=task.name, unique_config=task.unique_config, results=results)

        # take results from results store and continue
        else:
            Console.get_instance().log(f'Task {task.name}-{task.id_} already executed.')

    def _execute_task(self, task: Task):
        # extract predecessor results
        with self.lock:
            pred_results = self._extract_results_from_predecessors(task)
            self.running_queue.append(task.id_)

        # run the task and save results
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

                # TODO: Do we need a lock here?
                # run task only if it has not been executed already
                if task_id in self.done_queue or task_id in self.running_queue:
                    Console.get_instance().log(f'Task {task.name}-{task_id} is currently running or already finished.')
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
                Console.get_instance().log(f'Dolphin {self.id_} is now scheduling {successor.name}-{successor.id_}.')
                self.scheduled_queue.put(successor.id_)
