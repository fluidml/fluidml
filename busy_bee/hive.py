from multiprocessing import Manager, set_start_method, Queue, Event
from types import TracebackType
from typing import Optional, Type, List, Dict
from busy_bee.logging import Console

from busy_bee.task import Task
from busy_bee.bee import BusyBee, QueenBee


class Swarm:
    def __init__(self,
                 n_bees: int,
                 start_method: str = 'spawn',
                 refresh_every: int = 1):

        set_start_method(start_method, force=True)

        self.n_bees = n_bees
        self.manager = Manager()
        self.scheduled_queue = Queue()
        self.running_queue = self.manager.list()
        self.done_queue = self.manager.list()
        self.results = self.manager.dict()

        self.quit = Event()
        self.exception = self.manager.list()
        self.tasks: Dict[int, Task] = self.manager.dict()

        # queen bee for tracking
        self.busy_bees = [QueenBee(done_queue=self.done_queue,
                                   quit=self.quit,
                                   tasks=self.tasks,
                                   exception=self.exception,
                                   refresh_every=refresh_every)]
        # worker bees for task exection
        self.busy_bees.extend([BusyBee(bee_id=i,
                                       scheduled_queue=self.scheduled_queue,
                                       running_queue=self.running_queue,
                                       done_queue=self.done_queue,
                                       quit=self.quit,
                                       tasks=self.tasks,
                                       exception=self.exception,
                                       results=self.results) for i in range(self.n_bees)])

    def __enter__(self):
        return self

    def __exit__(self,
                 exc_type: Optional[Type[BaseException]],
                 exc_val: Optional[BaseException],
                 exc_tb: Optional[TracebackType]):
        self.close()

    @staticmethod
    def _get_entry_point_tasks(tasks):
        entry_task_ids = []
        for task in tasks:
            if len(task.predecessors) == 0:
                entry_task_ids.append(task.id_)
        return entry_task_ids

    def work(self, tasks: List[Task]):
        # get entry point task ids
        entry_point_task_ids = self._get_entry_point_tasks(tasks)

        # also update the current tasks
        for task in tasks:
            self.tasks[task.id_] = task

        # add entry point tasks to the job queue
        for task_id in entry_point_task_ids:
            Console.get_instance().log(f'Swarm scheduling {task_id}')
            self.scheduled_queue.put(task_id)

        # start the workers
        for bee in self.busy_bees:
            bee.start()

        # https://stackoverflow.com/questions/36962462/terminate-a-python-multiprocessing-program-once-a-one-of-its-workers-meets-a-cer
        # while not self.exception and len(self.done_queue) < len(self.tasks):

        # wait for them to finish
        for bee in self.busy_bees:
            bee.join()

        if self.quit.is_set():
            raise self.exception[0]

        return self.results

    def close(self):
        self.results = self.manager.dict()
        self.tasks = self.manager.dict()
        self.done_queue = self.manager.dict()
        for bee in self.busy_bees:
            bee.close()
