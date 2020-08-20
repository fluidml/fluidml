from multiprocessing import Process, Queue
from typing import Dict, Any
from hive.logging import Console
from rich.progress import Progress, BarColumn
import time


class BusyBee(Process):
    def __init__(self, bee_id: int, task_queue: Queue, done_queue, results_dict: Dict[str, Any]):
        super().__init__(target=self.work, args=(task_queue, done_queue, results_dict))
        self.bee_id = bee_id

    def work(self, task_queue: Queue, done_queue: Queue, results_dict: Dict[str, Any]):
        while True:
            # get the next task in the queue
            task = task_queue.get()

            if task == -1:  # end of the queue
                Console.get_instance().log(f"Bee {self.bee_id} leaving the swarm!")
                return

            Console.get_instance().log(f'Bee {self.bee_id} started running task {task.id}')

            # run the task
            results_dict[task.id] = task.run()

            Console.get_instance().log(f'Bee {self.bee_id} completed running task {task.id}')
            done_queue.put(task.id)


class QueenBee(Process):
    def __init__(self, task_queue: Queue, done_queue: Queue, refresh_every: int):
        self.refresh_every = refresh_every
        super().__init__(target=self.work, args=(task_queue, done_queue))

    def work(self, task_queue: Queue, done_queue: Queue):
        while task_queue.qsize() > 0:
            # sleep for a while
            time.sleep(1e-1)

            with Progress( "[progress.description]{task.description}", BarColumn(), "[progress.percentage]{task.percentage:>3.0f}%",) as progress:

                task = progress.add_task("[red]Task Progress...", total=task_queue.qsize() + done_queue.qsize())
                progress.update(task, advance=done_queue.qsize())

            # and show the stats
            Console.get_instance().log(f"Jobs in the task queue: {task_queue.qsize()}")
            Console.get_instance().log(f"Jobs in the done queue: {done_queue.qsize()}")
        return
