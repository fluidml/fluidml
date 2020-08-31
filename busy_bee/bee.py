from multiprocessing import Process, Queue, Lock
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
                 done_list: List,
                 id_to_task: Dict[int, Task],
                 graph: nx.DiGraph,
                 lock: Lock,
                 results: Dict[str, Any]):
        super().__init__(target=self.work,
                         args=())
        self.bee_id = bee_id
        self.task_queue = task_queue
        self.done_list = done_list
        self.id_to_task = id_to_task
        self.graph = graph
        self.results = results
        self.lock = lock

    def _is_task_ready(self,
                       task_id):
        dep_task_ids = list(self.graph.predecessors(task_id))

        for id_ in dep_task_ids:
            if id_ not in self.done_list:
                return False
        return True

    def work(self):
        while True:
            # terminate bee (worker) if all tasks have been processed
            # TODO: Does not work consistently
            if len(self.done_list) >= self.graph.number_of_nodes():  # end of the queue
                Console.get_instance().log(f'Bee {self.bee_id} leaving the swarm.')
                break

            # with self.lock:
            # get the next task in the queue
            task_id = self.task_queue.get()
            task = self.id_to_task[task_id]

            # run task only if all dependencies are satisfied
            if not self._is_task_ready(task.id_):
                continue

            # run task only if it has not been executed already
            if task_id in self.done_list:
                continue

            # run the task
            Console.get_instance().log(f'Bee {self.bee_id} started running task {task.id_}.')
            self.results[task.id_] = task.run()
            Console.get_instance().log(f'Bee {self.bee_id} completed running task {task.id_}.')

            # with self.lock:
            # put task in done_queue
            self.done_list.append(task.id_)

            # get successor tasks from graph and put them in task queue for processing
            successor_task_ids = list(self.graph.successors(task.id_))

            for id_ in successor_task_ids:
                self.task_queue.put(id_)


class QueenBee(Process):
    def __init__(self,
                 task_queue: Queue,
                 done_list: List,
                 graph: nx.DiGraph,
                 refresh_every: int):
        super().__init__(target=self.work, args=())

        self.task_queue = task_queue
        self.done_list = done_list
        self.graph = graph
        self.refresh_every = refresh_every

    def work(self):
        while len(self.done_list) < self.graph.number_of_nodes():
            # sleep for a while
            time.sleep(self.refresh_every)

            with Progress('[progress.description]{task.description}', BarColumn(),
                          '[progress.percentage]{task.percentage:>3.0f}%',) as progress:

                task = progress.add_task('[red]Task Progress...', total=self.graph.number_of_nodes())
                progress.update(task, advance=len(self.done_list))

            # and show the stats
            Console.get_instance().log(f'Jobs in the task queue: {self.task_queue.qsize()}')
            Console.get_instance().log(f'Jobs in the done queue: {len(self.done_list)}')
        return
