import logging
import logging.handlers
from multiprocessing import Queue
from threading import Thread

# from rich import console


class FluidLogger(Thread):
    def __init__(self, logging_queue: Queue):
        super().__init__(target=self.work,
                         args=())
        self.logging_queue = logging_queue

    def work(self):
        while True:
            record = self.logging_queue.get()
            if record is None:
                break
            logger = logging.getLogger(record.name)
            logger.handle(record)

# class Console:
#     __instance = None
#
#     @staticmethod
#     def get_instance():
#         """Static access method."""
#
#         if Console.__instance is None:
#             Console()
#         return Console.__instance
#
#     def __init__(self):
#         """Virtually private constructor."""
#
#         if Console.__instance is not None:
#             raise Exception('Use Console.get_instance()!')
#         else:
#             Console.__instance = console.Console()
