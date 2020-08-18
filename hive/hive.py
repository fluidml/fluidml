from hive.bee import BusyBee
from hive.task import Task
from multiprocessing import Queue
from typing import List


class Swarm:
    def __init__(self, n_bees: int):
        self.n_bees = n_bees
        self.task_queue = Queue()
        self.busy_bees = [BusyBee(i, self.task_queue) for i in range(self.n_bees)]

        # here, add a queen bee that tracks all other bees and tasks
        # if the queue is empty, queen bee leaves the swarm as well
        # but the queen bee logs all the messages

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

        # join
        for bee in self.busy_bees:
            bee.join()

        print("Swarm processing complete...")