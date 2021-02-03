import logging
from multiprocessing import Queue, Lock
from queue import Empty
from typing import Dict, Any, List, Optional, Tuple, Union

from fluidml.common import Task, Resource
from fluidml.swarm import Whale
from fluidml.storage import ResultsStore
from fluidml.storage.utils import pack_predecessor_results
from fluidml.common.utils import MyTask


logger = logging.getLogger(__name__)


class Dolphin(Whale):
    def __init__(self,
                 id_: int,
                 resource: Resource,
                 scheduled_queue: Queue,
                 running_queue: List[int],
                 done_queue: List[int],
                 logging_queue: Queue,
                 lock: Lock,
                 tasks: Dict[int, Task],
                 exception: Dict[str, Exception],
                 exit_on_error: bool,
                 results_store: Optional[ResultsStore] = None):
        super().__init__(exception=exception,
                         exit_on_error=exit_on_error,
                         logging_queue=logging_queue)
        self.id_ = id_
        self.resource = resource
        self.scheduled_queue = scheduled_queue
        self.running_queue = running_queue
        self.done_queue = done_queue
        self.lock = lock
        self.tasks = tasks
        self.results_store = results_store

    def _is_task_ready(self, task: Task):
        for predecessor in task.predecessors:
            if predecessor.id_ not in self.done_queue:
                return False
        return True

    def _extract_results_from_predecessors(self, task: Task) -> Dict[str, Any]:
        results: Dict = pack_predecessor_results(predecessor_tasks=task.predecessors,
                                                 results_store=self.results_store,
                                                 reduce_task=task.reduce,
                                                 task_expects=task.expects)
        return results

    def _run_task(self, task: Task, pred_results: Dict):
        # if force is set to false, try to get task results, else set results to none
        if task.force:
            results = None
        else:
            # try to get results from results store
            results: Optional[Tuple[Dict, str]
                              ] = self.results_store.get_results(task_name=task.name,
                                                                 task_unique_config=task.unique_config,
                                                                 task_publishes=task.publishes)
        # if results is none, run the task now
        if results is None:
            logger.info(f'Dolphin {self.id_} started running task {task.name}-{task.id_}.')
            if isinstance(task, MyTask):
                task.run(results=pred_results)
            else:
                task.run(**pred_results)
            logger.info(f'Dolphin {self.id_} completed running task {task.name}-{task.id_}.')

        # take results from results store and continue
        else:
            logger.info(f'Task {task.name}-{task.id_} already executed.')

    def _pack_task(self, task: Task) -> Task:
        task.results_store = self.results_store
        task.resource = self.resource
        return task

    def _execute_task(self, task: Task):
        # extract predecessor results
        with self.lock:
            pred_results = self._extract_results_from_predecessors(task)
            self.running_queue.append(task.id_)

        # pack the task
        task = self._pack_task(task)

        # run the task
        self._run_task(task, pred_results)

        with self.lock:
            # put task in done_queue
            self.done_queue.append(task.id_)
            logger.info(f'Finished {len(self.done_queue)} from {len(self.tasks)} tasks '
                        f'({round((len(self.done_queue) / len(self.tasks)) * 100)}%)')

    def _fetch_next_task(self) -> Union[int, None]:
        task_id = None
        try:
            task_id = self.scheduled_queue.get(block=False)
        except Empty:
            pass
        return task_id

    def _done(self) -> bool:
        return self.exception or (len(self.done_queue) == len(self.tasks))

    def _work(self):
        while not self._done():
            # fetch next task to run
            task_id = self._fetch_next_task()

            # continue when there is a valid task to run
            if task_id is not None:

                # get current task from task_id
                task = self.tasks[task_id]

                # TODO: Do we need a lock here? -> Yes, without I saw how 2 workers executed same task.
                with self.lock:
                    # run task only if it has not been executed already
                    if task_id in self.done_queue or task_id in self.running_queue:
                        logger.info(f'Task {task.name}-{task_id} is currently running or already finished.')
                        continue

                # all good to execute the task
                self._execute_task(task)

                # schedule the task's successors
                self._schedule_successors(task)

    def _schedule_successors(self, task: Task):
        # get successor tasks and put them in task queue for processing
        for successor in task.successors:
            # run task only if all dependencies are satisfied
            if not self._is_task_ready(task=self.tasks[successor.id_]):
                logger.info(f'Dolphin {self.id_}: Dependencies are not satisfied yet for '
                            f'task {successor.name}-{successor.id_}')
            elif successor.id_ in self.done_queue or successor.id_ in self.running_queue:
                logger.info(f'Task {successor.name}-{successor.id_} '
                            f'is currently running or already finished.')
            else:
                logger.info(f'Dolphin {self.id_} is now scheduling {successor.name}-{successor.id_}.')
                self.scheduled_queue.put(successor.id_)
