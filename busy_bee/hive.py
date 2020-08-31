from multiprocessing import Manager, set_start_method, Queue
from types import TracebackType
from typing import Optional, Type, List, Dict
from busy_bee.logging import Console

from busy_bee.task import Task
# from busy_bee.bee_queue import BeeQueue
from busy_bee.bee import BusyBee, QueenBee


class Swarm:
    def __init__(self,
                 n_bees: int,
                 tasks: List[Task],
                 start_method: str = 'spawn',
                 refresh_every: int = 1):

        set_start_method(start_method, force=True)

        self.n_bees = n_bees
        self.task_queue = Queue()
        self.manager = Manager()
        self.done_queue = self.manager.list()
        self.results = self.manager.dict()
        self.tasks: Dict[int, Task] = {task.id_: task for task in tasks}

        # queen bee for tracking
        self.busy_bees = [QueenBee(task_queue=self.task_queue,
                                   done_queue=self.done_queue,
                                   tasks=self.tasks,
                                   refresh_every=refresh_every)]
        # worker bees
        self.busy_bees.extend([BusyBee(bee_id=i,
                                       task_queue=self.task_queue,
                                       done_queue=self.done_queue,
                                       tasks=self.tasks,
                                       results=self.results) for i in range(self.n_bees)])

    def __enter__(self):
        return self

    def __exit__(self,
                 exc_type: Optional[Type[BaseException]],
                 exc_val: Optional[BaseException],
                 exc_tb: Optional[TracebackType]):
        self.close()

    def _get_entry_point_tasks(self):
        entry_task_ids = []
        for task_id, task in self.tasks.items():
            if len(task.pre_task_ids) == 0:
                entry_task_ids.append(task.id_)
        return entry_task_ids

    def work(self):
        # get entry point task ids
        entry_point_task_ids = self._get_entry_point_tasks()

        # TODO: add tasks dynamically
        # also update the current tasks
        #self.tasks = {task.id_: task for task in tasks}

        # add entry point tasks to the job queue
        for task_id in entry_point_task_ids:
            Console.get_instance().log(f'Swarm scheduling {task_id}')
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
        self.tasks = self.manager.dict()
        for bee in self.busy_bees:
            bee.terminate()
