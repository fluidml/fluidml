from abc import abstractmethod
from multiprocessing import Process
from typing import Dict


class Whale(Process):
    def __init__(self,
                 exception: Dict[str, Exception],
                 exit_on_error: bool):
        super().__init__(target=self.work,
                         args=())
        self.exception = exception
        self.exit_on_error = exit_on_error

    @abstractmethod
    def _work(self):
        raise NotImplementedError

    def work(self):
        try:
            self._work()
        except Exception as e:
            if self.exit_on_error:
                self.exception['message'] = e
            raise