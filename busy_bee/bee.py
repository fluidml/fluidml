from multiprocessing import Process
import time
from typing import Dict, Any

import networkx as nx
from rich.progress import Progress, BarColumn

from busy_bee import Task
from busy_bee.queue import Queue
from busy_bee.logging import Console


class BusyBee(Process):
    def __init__(self,
                 bee_id: int,
                 task_queue: Queue,
                 done_queue: Queue,
                 id_to_task: Dict[int, Task],
                 graph: nx.DiGraph,
                 results: Dict[str, Any]):
        super().__init__(target=self.work,
                         args=())
        self.bee_id = bee_id
        self.task_queue = task_queue
        self.done_queue = done_queue
        self.id_to_task = id_to_task
        self.graph = graph
        self.results = results

    def _is_task_ready(self,
                       task_id):
        dep_task_ids = list(self.graph.predecessors(task_id))

        for id_ in dep_task_ids:
            if id_ not in self.done_queue:
                return False
        return True

    def work(self):
        while True:
            # get the next task in the queue
            task_id = self.task_queue.get()
            task = self.id_to_task[task_id]

            # terminate bee (worker) if all tasks have been processed
            if self.done_queue.qsize() == self.graph.number_of_nodes():  # end of the queue
                Console.get_instance().log(f'Bee {self.bee_id} leaving the swarm.')
                return

            # run task only if all dependencies are satisfied
            if not self._is_task_ready(task.id_):
                continue

            # run the task
            Console.get_instance().log(f'Bee {self.bee_id} started running task {task.id_}.')
            self.results[task.id_] = task.run()
            Console.get_instance().log(f'Bee {self.bee_id} completed running task {task.id_}.')

            # put task in done_queue
            self.done_queue.put(task.id_)

            # get successor tasks from graph and put them in task queue for processing
            successor_task_ids = list(self.graph.successors(task.id_))
            for id_ in successor_task_ids:
                self.task_queue.put(id_)


class QueenBee(Process):
    def __init__(self,
                 task_queue: Queue,
                 done_queue: Queue,
                 graph: nx.DiGraph,
                 refresh_every: int):
        super().__init__(target=self.work, args=())

        self.task_queue = task_queue
        self.done_queue = done_queue
        self.graph = graph
        self.refresh_every = refresh_every

    def work(self):
        while self.done_queue.qsize() < self.graph.number_of_nodes():
            # sleep for a while
            time.sleep(self.refresh_every)

            with Progress('[progress.description]{task.description}', BarColumn(),
                          '[progress.percentage]{task.percentage:>3.0f}%',) as progress:

                task = progress.add_task('[red]Task Progress...', total=self.graph.number_of_nodes())
                progress.update(task, advance=self.done_queue.qsize())

            # and show the stats
            Console.get_instance().log(f'Jobs in the task queue: {self.task_queue.qsize()}')
            Console.get_instance().log(f'Jobs in the done queue: {self.done_queue.qsize()}')
        return
