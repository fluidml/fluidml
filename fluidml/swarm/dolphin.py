from collections import defaultdict
from multiprocessing import Queue, Lock
from queue import Empty
import time
from typing import Dict, Any, List, Optional, Tuple, Union

from rich.progress import Progress, BarColumn

from fluidml.common.task import Task, Resource
from fluidml.common.exception import TaskResultTypeError
from fluidml.common.logging import Console
from fluidml.swarm.storage import ResultsStorage
from fluidml.swarm.whale import Whale


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
                 results: Dict[str, Any],
                 results_storage: Optional[ResultsStorage] = None):
        super().__init__(exception=exception, exit_on_error=exit_on_error)
        self.id_ = id_
        self.resource = resource
        self.scheduled_queue = scheduled_queue
        self.running_queue = running_queue
        self.done_queue = done_queue
        self.lock = lock
        self.tasks = tasks
        self.results = results
        self.results_storage = results_storage

    def _is_task_ready(self, task: Task):
        for predecessor in task.predecessors:
            if predecessor.id_ not in self.done_queue:
                return False
        return True

    def _extract_results_from_predecessors(self, task: Task) -> Dict[str, Any]:
        results = {}
        for predecessor in task.predecessors:
            predecessor_result = self.results[predecessor.name][predecessor.id_]
            if isinstance(predecessor_result, dict):
                results = {**results,
                           **{f'{predecessor.name}-{predecessor.id_}':
                              predecessor_result}}
            else:
                raise TaskResultTypeError("Each task has to return a dict")

        return results

    def _extract_history_from_predecessors(self, task: Task) -> Dict[str, set]:
        history = defaultdict(set)
        for predecessor in task.predecessors:
            for name, pred_path in self.results[predecessor.name][predecessor.id_]['history'].items():
                history[name].update(pred_path)
        return history

    def _add_history_to_results_dict(self, history: Dict, task: Task):
        # Note: manager scripts can not be mutated, they have to be reassigned.
        #   see the first Note: https://docs.python.org/2/library/multiprocessing.html#managers

        if task.name not in self.results:
            self.results[task.name] = {}

        task_results = self.results[task.name]
        task_results[task.id_] = {'history': history}
        self.results[task.name] = task_results

    def _add_results_to_results_dict(self, results: Dict, task: Task):
        # Note: manager scripts can not be mutated, they have to be reassigned.
        #   see the first Note: https://docs.python.org/2/library/multiprocessing.html#managers

        if task.name not in self.results:
            self.results[task.name] = {}

        task_results = self.results[task.name]

        if task.id_ not in task_results:
            task_results[task.id_] = {}

        task_results[task.id_].update({'results': results,
                                       'config': task.unique_config})
        self.results[task.name] = task_results

    def _run_task_using_results_storage(self, task: Task, pred_results: Dict) -> Dict:
        with self.lock:
            history = self._extract_history_from_predecessors(task=task)

            # try to get results from results storage
            results: Optional[Tuple[Dict, str]] = self.results_storage.get_results(task_name=task.name,
                                                                                   unique_config=task.unique_config)
        # if results is none, it could not be retrieved -> run the task
        if results is None:
            Console.get_instance().log(f'Dolphin {self.id_} started running task {task.name}-{task.id_}.')
            results: Dict = task.run(results=pred_results, resource=self.resource)
            Console.get_instance().log(f'Dolphin {self.id_} completed running task {task.name}-{task.id_}.')

            # needs a lock to assure that for each task a unique storage path is created
            with self.lock:
                path = self.results_storage.save_results(task_name=task.name,
                                                         unique_config=task.unique_config,
                                                         results=results,
                                                         history=history)

        # if task.force = True -> run the task and overwrite existing results
        elif task.force:
            Console.get_instance().log(f'Dolphin {self.id_} started re-running task {task.name}-{task.id_}.')
            results: Dict = task.run(results=pred_results, resource=self.resource)
            Console.get_instance().log(f'Dolphin {self.id_} completed re-running task {task.name}-{task.id_}.')

            # needs a lock to assure that for each task a unique storage path is created
            with self.lock:
                path = self.results_storage.update_results(task_name=task.name,
                                                           unique_config=task.unique_config,
                                                           results=results,
                                                           history=history)

        # take results from results storage (unpack the tuple)
        else:
            with self.lock:
                results, path = results
                Console.get_instance().log(f'Task {task.name}-{task.id_} already executed.')

        with self.lock:
            # add task storage path to task history and add history to results dict
            history[task.name].add(path)
            self._add_history_to_results_dict(history=history, task=task)

        return results

    def _run_task(self, task: Task):
        # extract results from predecessors
        with self.lock:
            pred_results = self._extract_results_from_predecessors(task)

            # add to list of running tasks
            self.running_queue.append(task.id_)

        if self.results_storage:
            # run task using file storage for versioning
            results: Dict = self._run_task_using_results_storage(task=task, pred_results=pred_results)
        else:
            # run task only
            Console.get_instance().log(f'Dolphin {self.id_} started running task {task.name}-{task.id_}.')
            results: Dict = task.run(results=pred_results, resource=self.resource)
            Console.get_instance().log(f'Dolphin {self.id_} completed running task {task.name}-{task.id_}.')

        with self.lock:
            # add results to shared results dict
            self._add_results_to_results_dict(results=results, task=task)

            # put task in done_queue
            self.done_queue.append(task.id_)

    def _fetch_next_task(self) -> Union[str, None]:
        task_id = None
        try:
            task_id = self.scheduled_queue.get(block=False, timeout=0.5)
        except Empty:
            Console.get_instance().log(f'Bee {self.id_}: waiting for tasks.')
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

                # all good to run the task
                self._run_task(task)

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
                Console.get_instance().log(f'Task {successor.name}-{successor.id_} is currently running or already finished.')
            else:
                Console.get_instance().log(f'Dolphin {self.id_} is now scheduling {successor.name}-{successor.id_}.')
                self.scheduled_queue.put(successor.id_)
