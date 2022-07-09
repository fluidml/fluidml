import logging
from multiprocessing import Queue, Lock, Event
from queue import Empty
from typing import Dict, Any, List, Union

from fluidml.common import Task, Resource
from fluidml.flow.task_spec import TaskSpec
from fluidml.storage.controller import TaskDataController
from fluidml.swarm import Whale

logger = logging.getLogger(__name__)


class Dolphin(Whale):
    def __init__(
        self,
        resource: Resource,
        scheduled_queue: Queue,
        running_queue: List[int],
        done_queue: List[int],
        logging_queue: Queue,
        error_queue: Queue,
        lock: Lock,
        tasks: Dict[int, TaskSpec],
        exit_event: Event,
        exit_on_error: bool,
        logging_lvl: int,
    ):
        super().__init__(
            exit_event=exit_event,
            exit_on_error=exit_on_error,
            logging_queue=logging_queue,
            error_queue=error_queue,
            logging_lvl=logging_lvl,
            lock=lock,
        )
        self.resource = resource
        self.scheduled_queue = scheduled_queue
        self.running_queue = running_queue
        self.done_queue = done_queue
        self.tasks = tasks

    @property
    def num_tasks(self):
        return len(self.tasks)

    def _is_task_ready(self, task_spec: TaskSpec):
        for predecessor in task_spec.predecessors:
            if predecessor.id_ not in self.done_queue:
                return False
        return True

    def _extract_results_from_predecessors(self, task: Task) -> Dict[str, Any]:
        with self._lock:
            controller = TaskDataController(task)
            results: Dict = controller.pack_predecessor_results()
        return results

    def _run_task(self, task: Task):
        # if force is true, delete all task results and re-run task
        if task.force:
            task.delete_run()

        with self._lock:
            # check if task was successfully completed before
            completed: bool = task.results_store.is_finished(task_name=task.name, task_unique_config=task.unique_config)
        # if task is not completed, run the task now
        if not completed:
            # extract predecessor results
            pred_results = self._extract_results_from_predecessors(task)

            logger.info(f"Started task {task.unique_name}.")
            task.run_wrapped(**pred_results)

        with self._lock:
            # put task in done_queue
            self.done_queue.append(task.id_)

            # remove task from running_queue
            idx = self.running_queue.index(task.id_)
            del self.running_queue[idx]

            # Log task completion
            if completed:
                msg = f"Task {task.unique_name} already executed"
            else:
                msg = f"Finished task {task.unique_name}"

            logger.info(
                f"{msg} [{len(self.done_queue)}/{self.num_tasks} "
                f"- {round((len(self.done_queue) / self.num_tasks) * 100)}%]"
            )

    def _pack_task(self, task: Task) -> Task:
        task.resource = self.resource
        task.lock = self._lock
        return task

    def _execute_task(self, task: Task):
        # pack the task
        task = self._pack_task(task)

        # run the task
        self._run_task(task)

    def _fetch_next_task(self) -> Union[int, None]:
        try:
            task_id = self.scheduled_queue.get(block=False, timeout=0.5)
        except Empty:
            task_id = None
        return task_id

    def _done(self) -> bool:
        return self.exit_event.is_set() or (len(self.done_queue) == len(self.tasks))

    def _work(self):
        while not self._done():
            # fetch next task to run
            task_id = self._fetch_next_task()

            # continue when there is a valid task to run
            if task_id is not None:

                # get current task_spec from task_id
                task_spec = self.tasks[task_id]

                # instantiate task obj
                task = Task.from_spec(task_spec)

                # execute the task
                self._execute_task(task)

                with self._lock:
                    # schedule the task's successors
                    self._schedule_successors(task_spec)

    def _schedule_successors(self, task_spec: TaskSpec):
        # get successor tasks and put them in task queue for processing
        for successor in task_spec.successors:
            # run task only if all dependencies are satisfied
            if not self._is_task_ready(task_spec=self.tasks[successor.id_]):
                logger.debug(f"Dependencies are not satisfied yet for " f"task {successor.unique_name}")
            # the done_queue check should not be necessary because tasks don't leave the running queue once they're
            # finished. we use the done_queue for progress measuring and running_queue to avoid tasks being executed
            # twice.
            elif successor.id_ in self.done_queue or successor.id_ in self.running_queue:
                logger.debug(f"Task {successor.unique_name} " f"is currently running or already finished.")
            else:
                logger.debug(f"Is now scheduling {successor.unique_name}.")
                self.scheduled_queue.put(successor.id_)
                # We have to add the successor id to the running queue here already
                # Assume, 2 workers execute a task each in parallel, finish at the same time
                # and both tasks have the same successor task.
                # Due to the lock only 1 worker enters this fn and puts his task's successor ids
                # in the scheduled queue. Once the worker leaves this fn the lock gets released
                # and the second worker adds his task's successors to the queue.
                # Now, if the shared successor id hasn't been picked up yet by another worker
                # which previously put the id in the running queue, the second worker puts
                # the successor id again in the schedule queue.
                # That leads to the task being executed more than once.
                # Hence, we have to add the successor ids to the running queue in the moment they are
                # added to the schedule queue.
                self.running_queue.append(successor.id_)
