from multiprocessing import Process, Lock, Queue
from queue import Empty
import time
from typing import Dict, Any, List

import networkx as nx
from rich.progress import Progress, BarColumn

from busy_bee.task import Task
# from busy_bee.bee_queue import BeeQueue
from busy_bee.logging import Console


class BusyBee(Process):
    def __init__(self,
                 bee_id: int,
                 task_queue: Queue,
                 done_queue: List[int],
                 tasks: Dict[str, Task],
                 results: Dict[str, Any]):
        super().__init__(target=self.work,
                         args=(task_queue, done_queue, tasks, results))
        self.bee_id = bee_id

    @staticmethod
    def _is_task_ready(task: Task, done_queue: List[int]):
        dep_task_ids = task.pre_task_ids

        for id_ in dep_task_ids:
            if id_ not in done_queue:
                return False
        return True

    def work(self, task_queue: Queue, done_queue: Queue, tasks: Dict[str, Task], results: Dict[str, Any]):
        while True:
            # terminate bee (worker) if all tasks have been processed
            # TODO: Seems to work, test more
            if len(done_queue) >= len(tasks):  # end of the queue
                Console.get_instance().log(f'Bee {self.bee_id} leaving the swarm.')
                break

            # get the next task in the queue
            try:
                task_id = task_queue.get(block=False, timeout=5)
            except Empty:
                if len(done_queue) == len(tasks):
                    Console.get_instance().log(f'Bee {self.bee_id}: leaving the swarm.')
                    break
                else:
                    Console.get_instance().log(f'Bee {self.bee_id}: is retrying.')
                    continue
            task = tasks[task_id]

            # run task only if all dependencies are satisfied
            if not BusyBee._is_task_ready(task, done_queue):
                Console.get_instance().log(f'Bee {self.bee_id}: Dependencies are not satisfied yet for task {task.id_}')
                continue

            # run task only if it has not been executed already
            if task_id in done_queue:
                Console.get_instance().log(f'{task_id} already finished')
                continue

            # run the task
            Console.get_instance().log(f'Bee {self.bee_id} started running task {task.id_}.')
            results[task.id_] = task.run()
            Console.get_instance().log(f'Bee {self.bee_id} completed running task {task.id_}.')

            # put task in done_queue
            done_queue.append(task.id_)

            # get successor tasks and put them in task queue for processing
            for id_ in task.post_task_ids:
                Console.get_instance().log(f'Bee {self.bee_id} is now scheduling {id_}.')
                task_queue.put(id_)


class QueenBee(Process):
    def __init__(self,
                 task_queue: Queue,
                 done_queue: List[int],
                 tasks: Dict[str, Task],
                 refresh_every: int):
        super().__init__(target=self.work, args=(task_queue, done_queue, tasks))
        self.refresh_every = refresh_every

    def work(self, task_queue: Queue, done_queue: Queue, tasks: Dict[str, Task]):
        while len(done_queue) < len(tasks):
            # sleep for a while
            time.sleep(self.refresh_every)

            with Progress('[progress.description]{task.description}', BarColumn(),
                          '[progress.percentage]{task.percentage:>3.0f}%',) as progress:

                task = progress.add_task('[red]Task Progress...', total=len(tasks))
                progress.update(task, advance=len(done_queue))

            # and show the stats
            Console.get_instance().log(f'Jobs in the task queue: {task_queue.qsize()}')
            Console.get_instance().log(f'Jobs in the done queue: {len(done_queue)}')
        return
