from hive.bee import BusyBee, QueenBee
from hive.task import Task
from multiprocessing import Queue, Manager
from typing import List


class Swarm:
    def __init__(self, n_bees: int, refresh_every: int = 1):
        self.n_bees = n_bees
        self.task_queue = Queue()
        self.done_queue = Queue()
        self.manager = Manager()
        self.results_dict = self.manager.dict()
        self.busy_bees = [QueenBee(self.task_queue, self.done_queue, refresh_every)]  # queen bee for  tracking
        self.busy_bees.extend([BusyBee(i, self.task_queue, self.done_queue, self.results_dict) for i in range(self.n_bees)])

    def work(self, tasks: List[Task]):

        # add task to the job queue
        for task in tasks:
            self.task_queue.put(task)

        # append terminate
        for i in range(self.n_bees):
            self.task_queue.put(-1)

        # start the workers
        for bee in self.busy_bees:
            bee.start()

        # wait for them to finish
        for bee in self.busy_bees:
            bee.join()

        return self.results_dict

    def close(self):
        self.results_dict = self.manager.dict()
        for bee in self.busy_bees:
            bee.terminate()
