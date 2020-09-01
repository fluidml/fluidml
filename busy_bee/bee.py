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
                 task_queue: Queue,
                 done_queue: List[int],
                 tasks: Dict[str, Task],
                 results: Dict[str, Any]):
        super().__init__(target=self.work,
                         args=())
        self.bee_id = bee_id
        self.task_queue = task_queue
        self.done_queue = done_queue
        self.tasks = tasks
        self.results = results

    def _is_task_ready(self, task: Task):
        dep_task_ids = task.pre_task_ids

        for id_ in dep_task_ids:
            if id_ not in self.done_queue:
                return False
        return True

    def work(self):
        while True:
            # TODO: Seems to work, test more
            # get the next task in the queue
            try:
                task_id = self.task_queue.get(block=False, timeout=1)
            except Empty:
                # terminate bee (worker) if all tasks have been processed
                if len(self.done_queue) == len(self.tasks):
                    Console.get_instance().log(f'Bee {self.bee_id}: leaving the swarm.')
                    break
                else:
                    Console.get_instance().log(f'Bee {self.bee_id}: is retrying.')
                    # break
                    continue
            task = self.tasks[task_id]

            # run task only if all dependencies are satisfied
            if not self._is_task_ready(task=task):
                Console.get_instance().log(f'Bee {self.bee_id}: Dependencies are not satisfied yet for task {task.id_}')
                continue

            # TODO: Do we need a lock here?
            # run task only if it has not been executed already
            if task_id in self.done_queue:
                Console.get_instance().log(f'Task {task_id} already finished.')
                continue

            # put task in done_queue
            self.done_queue.append(task.id_)
            # TODO: until here?

            # run the task
            Console.get_instance().log(f'Bee {self.bee_id} started running task {task.id_}.')
            self.results[task.id_] = task.run()
            Console.get_instance().log(f'Bee {self.bee_id} completed running task {task.id_}.')

            # get successor tasks and put them in task queue for processing
            for id_ in task.post_task_ids:
                Console.get_instance().log(f'Bee {self.bee_id} is now scheduling {id_}.')
                self.task_queue.put(id_)


class QueenBee(Process):
    def __init__(self,
                 task_queue: Queue,
                 done_queue: List[int],
                 tasks: Dict[str, Task],
                 refresh_every: int):
        super().__init__(target=self.work, args=())
        self.refresh_every = refresh_every
        self.task_queue = task_queue
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

            # and show the stats
            Console.get_instance().log(f'Jobs in the task queue: {self.task_queue.qsize()}')
            Console.get_instance().log(f'Jobs in the done queue: {len(self.done_queue)}')
