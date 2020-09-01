from multiprocessing import Process, Queue
from queue import Empty
import time
from typing import Dict, Any, List

from rich.progress import Progress, BarColumn

from busy_bee.task import Task
from busy_bee.logging import Console


class BusyBee(Process):
    def __init__(self,
                 bee_id: int,
                 scheduled_queue: Queue,
                 running_queue: List[int],
                 done_queue: List[int],
                 tasks: Dict[str, Task],
                 results: Dict[str, Any]):
        super().__init__(target=self.work,
                         args=())
        self.bee_id = bee_id
        self.scheduled_queue = scheduled_queue
        self.running_queue = running_queue
        self.done_queue = done_queue
        self.tasks = tasks
        self.results = results

    def _is_task_ready(self, task: Task):
        for id_ in task.predecessors:
            if id_ not in self.done_queue:
                return False
        return True

    def _extract_results_from_predecessors(self, task: Task) -> Dict[str, Any]:
        results = {}
        for predecessor in task.predecessors:
            results = {**results, **self.results[predecessor]}
        return results

    def _run_task(self, task: Task):
        # extract results from predecessors
        results = self._extract_results_from_predecessors(task)

        # add to list of running tasks
        self.running_queue.append(task.id_)

        # run task
        Console.get_instance().log(f'Bee {self.bee_id} started running task {task.id_}.')
        self.results[task.id_] = task.run(results)
        Console.get_instance().log(f'Bee {self.bee_id} completed running task {task.id_}.')

        # put task in done_queue
        self.done_queue.append(task.id_)

    def work(self):
        while True:
            # TODO: Seems to work, test more
            # get the next task in the queue
            try:
                task_id = self.scheduled_queue.get(block=False, timeout=0.5)
            except Empty:
                # terminate bee (worker) if all tasks have been processed
                if len(self.done_queue) == len(self.tasks):
                    Console.get_instance().log(f'Bee {self.bee_id}: leaving the swarm.')
                    break
                else:
                    #Console.get_instance().log(f'Bee {self.bee_id}: waiting for tasks.')
                    # break
                    continue

            # current task
            task = self.tasks[task_id]

            # run task only if all dependencies are satisfied
            if not self._is_task_ready(task=task):
                Console.get_instance().log(f'Bee {self.bee_id}: Dependencies are not satisfied yet for task {task.id_}')
                continue

            # TODO: Do we need a lock here?
            # run task only if it has not been executed already
            if task_id in self.done_queue or task_id in self.running_queue:
                Console.get_instance().log(f'Task {task_id} is currently running or already finished.')
                continue

            # all good to run the task
            self._run_task(task)

            # get successor tasks and put them in task queue for processing
            for id_ in task.successors:
                Console.get_instance().log(f'Bee {self.bee_id} is now scheduling {id_}.')
                self.scheduled_queue.put(id_)


class QueenBee(Process):
    def __init__(self,
                 done_queue: List[int],
                 tasks: Dict[str, Task],
                 refresh_every: int):
        super().__init__(target=self.work, args=())
        self.refresh_every = refresh_every
        self.done_queue = done_queue
        self.tasks = tasks

    def work(self):
        while len(self.done_queue) < len(self.tasks):
            # sleep for a while
            time.sleep(self.refresh_every)

            with Progress('[progress.description]{task.description}', BarColumn(),
                          '[progress.percentage]{task.percentage:>3.0f}%',) as progress:

                task = progress.add_task('[red]Task Progress...', total=len(self.tasks))
                progress.update(task, advance=len(self.done_queue))
