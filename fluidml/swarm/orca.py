import logging
from multiprocessing import Queue
import time
from typing import Dict, List

# from rich.progress import Progress, BarColumn

from fluidml.common import Task
from fluidml.swarm import Whale


class Orca(Whale):
    def __init__(self,
                 done_queue: List[int],
                 logging_queue: Queue,
                 exception: Dict[str, Exception],
                 exit_on_error: bool,
                 tasks: Dict[int, Task],
                 refresh_every: int):
        super().__init__(exception=exception, exit_on_error=exit_on_error, logging_queue=logging_queue)
        self.refresh_every = refresh_every
        self.done_queue = done_queue
        self.tasks = tasks

    def _work(self):
        logger = logging.getLogger(__name__)
        while len(self.done_queue) < len(self.tasks) and not self.exception:
            if self.refresh_every is not None:
                # sleep for a while
                time.sleep(self.refresh_every)
                logger.info(f'Finished {len(self.done_queue)} from {len(self.tasks)} tasks '
                            f'({round((len(self.done_queue) / len(self.tasks)) * 100)}%)')

                # with Progress('[progress.description]{task.description}', BarColumn(),
                #               '[progress.percentage]{task.percentage:>3.0f}%',) as progress:
                #
                #     task = progress.add_task('[red]Task Progress...', total=len(self.tasks))
                #     progress.update(task, advance=len(self.done_queue))
