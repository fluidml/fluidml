from abc import abstractmethod
from multiprocessing import Process, Queue, Lock
from queue import Empty
import time
from typing import Dict, Any, List, Optional, Tuple

from rich.progress import Progress, BarColumn

from fluidml.common.task import Task, Resource
from fluidml.common.logging import Console
from fluidml.swarm.storage import ResultsStorage


class Whale(Process):
    def __init__(self,
                 exception: Dict[str, Exception],
                 exit_on_error: bool):
        super().__init__(target=self.work,
                         args=())
        self.exception = exception
        self.exit_on_error = exit_on_error

    @abstractmethod
    def _work(self):
        raise NotImplementedError

    def work(self):
        try:
            self._work()
        except Exception as e:
            if self.exit_on_error:
                self.exception['message'] = e
            raise


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
            try:
                results = {**results, **{predecessor.name: self.results[predecessor.name][predecessor.id_]['results']}}
            except TypeError:
                print('Each task has to return a dict.')
                raise

        return results

    def _add_results_to_results_dict(self, results: Dict, task: Task):
        # Note: manager dicts can not be mutated, they have to be reassigned.
        #   see the first Note: https://docs.python.org/2/library/multiprocessing.html#managers

        if task.name not in self.results:
            self.results[task.name] = {}

        task_results = self.results[task.name]
        task_results[task.id_] = {'results': results,
                                  'config': task.unique_config}
        self.results[task.name] = task_results

    def _run_task_using_results_storage(self, task: Task, pred_results: Dict) -> Dict:
        if self.tasks[task.id_].storage_path is None:
            task.storage_path = {}
            self.tasks[task.id_] = task

        # for all predecessor tasks write their storage path history in the current task storage path
        for predecessor in task.predecessors:
            for name, pred_path in self.tasks[predecessor.id_].storage_path.items():
                task.storage_path[name] = pred_path
        self.tasks[task.id_] = task

        results: Optional[Tuple[Dict, str]] = self.results_storage.get_results(task=task)

        if results is None:
            # run task
            Console.get_instance().log(f'Dolphin {self.id_} started running task {task.name}-{task.id_}.')
            results: Dict = task.run(pred_results, self.resource)
            Console.get_instance().log(f'Dolphin {self.id_} completed running task {task.name}-{task.id_}.')

            # TODO: This probably needs a lock
            with self.lock:
                path = self.results_storage.save_results(task=task, results=results)
        else:
            results, path = results
            Console.get_instance().log(f'Task {task.name}-{task.id_} already executed.')

        # Write unique storage path (e.g. run_dir) of task to its task object
        task.storage_path[task.name] = path
        self.tasks[task.id_] = task
        return results

    def _run_task(self, task: Task):
        # extract results from predecessors
        pred_results = self._extract_results_from_predecessors(task)

        # add to list of running tasks
        self.running_queue.append(task.id_)

        if self.results_storage:
            # run task using file storage for versioning
            results: Dict = self._run_task_using_results_storage(task=task, pred_results=pred_results)
        else:
            # run task only
            Console.get_instance().log(f'Dolphin {self.id_} started running task {task.name}-{task.id_}.')
            results: Dict = task.run(pred_results, self.resource)
            Console.get_instance().log(f'Dolphin {self.id_} completed running task {task.name}-{task.id_}.')

        # add results to shared results dict
        self._add_results_to_results_dict(results=results, task=task)

        # put task in done_queue
        self.done_queue.append(task.id_)

    def _work(self):
        while not self.exception:
            # TODO: Seems to work, test more
            # get the next task in the queue
            try:
                task_id = self.scheduled_queue.get(block=False, timeout=0.5)
            except Empty:
                # terminate bee (worker) if all tasks have been processed
                if len(self.done_queue) == len(self.tasks):
                    Console.get_instance().log(f'Dolphin {self.id_}: leaving the swarm.')
                    break
                else:
                    # Console.get_instance().log(f'Bee {self.bee_id}: waiting for tasks.')
                    continue

            # get current task from task_id
            task = self.tasks[task_id]

            # TODO: Do we need a lock here?
            # run task only if it has not been executed already
            if task_id in self.done_queue or task_id in self.running_queue:
                Console.get_instance().log(f'Task {task.name}-{task_id} is currently running or already finished.')
                continue

            # all good to run the task
            self._run_task(task)

            # get successor tasks and put them in task queue for processing
            for successor in task.successors:
                # run task only if all dependencies are satisfied
                if not self._is_task_ready(task=self.tasks[successor.id_]):
                    Console.get_instance().log(
                        f'Dolphin {self.id_}: Dependencies are not satisfied yet for '
                        f'task {successor.name}-{successor.id_}')
                    continue

                if successor.id_ in self.done_queue or successor.id_ in self.running_queue:
                    Console.get_instance().log(f'Task {successor.name}-{successor.id_} '
                                               f'is currently running or already finished.')
                    continue

                Console.get_instance().log(f'Dolphin {self.id_} is now scheduling {successor.name}-{successor.id_}.')
                self.scheduled_queue.put(successor.id_)


class Orca(Whale):
    def __init__(self,
                 done_queue: List[int],
                 exception: Dict[str, Exception],
                 exit_on_error: bool,
                 tasks: Dict[int, Task],
                 refresh_every: int):
        super().__init__(exception=exception, exit_on_error=exit_on_error)
        self.refresh_every = refresh_every
        self.done_queue = done_queue
        self.tasks = tasks

    def _work(self):
        while len(self.done_queue) < len(self.tasks) and not self.exception:
            # sleep for a while
            time.sleep(self.refresh_every)

            with Progress('[progress.description]{task.description}', BarColumn(),
                          '[progress.percentage]{task.percentage:>3.0f}%',) as progress:

                task = progress.add_task('[red]Task Progress...', total=len(self.tasks))
                progress.update(task, advance=len(self.done_queue))
