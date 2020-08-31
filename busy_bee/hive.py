from multiprocessing import Manager, set_start_method, Queue, Lock
from types import TracebackType
from typing import Optional, Type, Dict


import networkx as nx

from busy_bee.task import Task
# from busy_bee.bee_queue import BeeQueue
from busy_bee.bee import BusyBee, QueenBee


class Swarm:
    def __init__(self,
                 n_bees: int,
                 graph: nx.DiGraph,
                 id_to_task: Dict[int, Task],
                 start_method: str = 'spawn',
                 refresh_every: int = 1):

        set_start_method(start_method, force=True)

        self.n_bees = n_bees
        self.graph = graph
        self.id_to_task = id_to_task
        self.task_queue = Queue()
        # self.done_queue = Queue()
        self.manager = Manager()
        self.results = self.manager.dict()
        self.done_list = self.manager.list()
        self.lock = Lock()

        # queen bee for tracking
        self.busy_bees = [QueenBee(task_queue=self.task_queue,
                                   done_list=self.done_list,
                                   graph=self.graph,
                                   refresh_every=refresh_every)]
        # worker bees
        self.busy_bees.extend([BusyBee(bee_id=i,
                                       graph=self.graph,
                                       id_to_task=self.id_to_task,
                                       task_queue=self.task_queue,
                                       done_list=self.done_list,
                                       lock=self.lock,
                                       results=self.results) for i in range(self.n_bees)])

    def __enter__(self):
        return self

    def __exit__(self,
                 exc_type: Optional[Type[BaseException]],
                 exc_val: Optional[BaseException],
                 exc_tb: Optional[TracebackType]):
        self.close()

    def _get_entry_point_tasks(self):
        task_ids = []
        for node in self.graph.nodes:
            if len(list(self.graph.predecessors(node))) == 0:
                task_ids.append(node)
        assert len(task_ids) > 0, 'The dependency graph does not have any entry-point nodes.'
        return task_ids

    def work(self):
        entry_point_task_ids = self._get_entry_point_tasks()

        # add entry point tasks to the job queue
        for task_id in entry_point_task_ids:
            self.task_queue.put(task_id)

        # start the workers
        for bee in self.busy_bees:
            bee.start()

        # wait for them to finish
        for bee in self.busy_bees:
            bee.join()

        return self.results

    def close(self):
        self.results = self.manager.dict()
        self.done_list = self.manager.list()
        for bee in self.busy_bees:
            bee.terminate()
