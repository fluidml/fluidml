import logging
from multiprocessing import Queue, Lock, Event
from queue import Empty
from typing import Dict, Any, List, Union, Optional

from fluidml.common import Task
from fluidml.common.utils import change_logging_level
from fluidml.flow.task_spec import TaskSpec
from fluidml.storage.controller import TaskDataController
from fluidml.swarm import Whale

# from fluidml.common.task import TaskState

logger = logging.getLogger(__name__)


class Dolphin(Whale):
    def __init__(
        self,
        scheduled_queue: Queue,
        # task_states: Dict,
        running_queue: List[str],
        done_queue: List[str],
        failed_queue: List[str],
        logging_queue: Queue,
        error_queue: Queue,
        lock: Lock,
        tasks: Dict[str, TaskSpec],
        exit_event: Event,
        exit_on_error: bool,
        logging_lvl: int,
        resource: Optional[Any] = None,
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
        # self.task_states = task_states
        self.running_queue = running_queue
        self.done_queue = done_queue
        self.failed_queue = failed_queue
        self.tasks = tasks

    @property
    def num_tasks(self):
        return len(self.tasks)

    def _is_task_ready(self, task_spec: TaskSpec):
        for predecessor in task_spec.predecessors:
            # if predecessor.status != Status.FINISHED:
            # if self.task_states[predecessor.unique_name] != Status.FINISHED:
            if predecessor.unique_name not in self.done_queue:
                return False
        return True

    def _execute_task(self, task: Task):

        # provide resource and lock
        task.resource = self.resource
        task.results_store.lock = self._lock

        # run the task
        completed: bool = run_task(task)

        with self._lock:
            # self.tasks[task.unique_name].status = Status.FINISHED
            # self.task_states[task.unique_name] = Status.FINISHED
            # put task in done_queue
            self.done_queue.append(task.unique_name)

            # remove task from running_queue
            idx = self.running_queue.index(task.unique_name)
            del self.running_queue[idx]

            # Log task completion
            if completed:
                msg = f'Task "{task.unique_name}" already executed'
            else:
                msg = f'Finished task "{task.unique_name}"'

            # num_finished_task = sum(1 for t in self.task_states.values() if t == Status.FINISHED)
            # # num_finished_task = sum(1 for t in self.tasks.values() if t.status == Status.FINISHED)
            # logger.info(
            #     f"{msg} [{num_finished_task}/{self.num_tasks} "
            #     f"- {round((num_finished_task / self.num_tasks) * 100)}%]"
            # )
            logger.info(
                f"{msg} [{len(self.done_queue)}/{self.num_tasks} "
                f"- {round((len(self.done_queue) / self.num_tasks) * 100)}%]"
            )

    def _fetch_next_task(self) -> Union[str, None]:
        try:
            task_counter = self.scheduled_queue.get(block=False, timeout=0.5)
        except Empty:
            task_counter = None
        return task_counter

    def _done(self) -> bool:
        # print((len(self.done_queue) + len(self.failed_queue)))
        # num_finished_task = sum(1 for t in self.tasks.values() if t.status == Status.FINISHED)
        # s = sum(True if t.status in [Status.FINISHED, Status.FAILED] else False for t in self.tasks.values())
        # print(num_finished_task)
        return (
            self.exit_event.is_set()
            # or all(True if t.status in [Status.FINISHED, Status.FAILED] else False for t in self.tasks.values())
            # or all(True if t in [Status.FINISHED, Status.FAILED] else False for t in self.task_states.values())
            or len(self.done_queue) == len(self.tasks)
            # or (len(self.done_queue) + len(self.failed_queue) == len(self.tasks))
        )

    def _work(self):
        while not self._done():
            # fetch next task to run
            task_unique_name = self._fetch_next_task()

            # continue when there is a valid task to run
            if task_unique_name is not None:
                # get current task_spec from task_unique_name
                task_spec = self.tasks[task_unique_name]

                # get current task_spec from task_unique_name
                self.running_queue.append(task_spec.unique_name)

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
            if not self._is_task_ready(task_spec=self.tasks[successor.unique_name]):
                logger.debug(f"Dependencies are not satisfied yet for " f"task {successor.unique_name}")
            # the done_queue check should not be necessary because tasks don't leave the running queue once they're
            # finished. we use the done_queue for progress measuring and running_queue to avoid tasks being executed
            # twice.
            elif successor.unique_name in self.done_queue or successor.unique_name in self.running_queue:
                logger.debug(f"Task {successor.unique_name} " f"is currently running or already finished.")
            else:
                logger.debug(f"Is now scheduling {successor.unique_name}.")
                self.scheduled_queue.put(successor.unique_name)
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
                # Hence, we have to add the successor ids to the running queue at the moment they are
                # added to the schedule queue.
                self.running_queue.append(successor.unique_name)

    # def _work(self):
    #
    #     while not self._done():
    #
    #         # fetch next task to run
    #         task_unique_name = self._fetch_next_task()
    #
    #         # continue when there is a valid task to run
    #         if task_unique_name is not None:
    #             # get current task_spec from task_unique_name
    #             task_spec = self.tasks[task_unique_name]
    #
    #             try:
    #                 with self._lock:
    #                     # check if any predecessor task is in failed_tasks list
    #                     failed_predecessor = None
    #                     for predecessor in task_spec.predecessors:
    #                         # if predecessor.unique_name in self.failed_queue:
    #                         # if predecessor.status == Status.FAILED:
    #                         if self.task_states[predecessor.unique_name] == Status.FAILED:
    #                             failed_predecessor = predecessor.unique_name
    #                             break
    #
    #                     if failed_predecessor:
    #                         logger.info(
    #                             f'Task "{task_unique_name}" cannot be executed since predecessor task "{failed_predecessor}" failed.'
    #                         )
    #                         # if task_spec.unique_name not in self.failed_queue:
    #                         #     self.failed_queue.append(task_spec.unique_name)
    #                         self.task_states[task_spec.unique_name] = Status.FAILED
    #                         # task_spec.status = Status.FAILED
    #
    #                         self._schedule_successors(task_spec)
    #                         continue
    #
    #                     # run task only if all dependencies are satisfied
    #                     if not self._is_task_ready(task_spec=task_spec):
    #                         logger.debug(f"Dependencies are not satisfied yet for " f"task {task_spec.unique_name}")
    #                         self.scheduled_queue.put(task_spec.unique_name)
    #                         continue
    #
    #                     # self.running_queue.append(task_spec.unique_name)
    #
    #                 # instantiate task obj
    #                 task = Task.from_spec(task_spec)
    #
    #                 self.task_states[task.unique_name] = Status.RUNNING
    #
    #                 task.status = self.task_states[task.unique_name]
    #                 # execute the task
    #                 self._execute_task(task)
    #
    #                 with self._lock:
    #                     # schedule the task's successors
    #                     self._schedule_successors(task_spec)
    #             except Exception as e:
    #                 logger.info(f'Task "{task_unique_name}" cannot be executed.')
    #                 # if task_unique_name not in self.failed_queue:
    #                 #     self.failed_queue.append(task_unique_name)
    #                 self.task_states[task_unique_name] = Status.FAILED
    #                 # task_spec.status = Status.FAILED
    #                 self._schedule_successors(task_spec)
    #                 raise
    #
    # def _schedule_successors(self, task_spec: TaskSpec):
    #     # get successor tasks and put them in task queue for processing
    #     for successor in task_spec.successors:
    #         # if successor.status == Status.CUED:
    #         if self.task_states[successor.unique_name] == Status.CUED:
    #             logger.debug(f"Is now scheduling {successor.unique_name}.")
    #             self.scheduled_queue.put(successor.unique_name)
    #             self.task_states[successor.unique_name] = Status.SCHEDULED
    #             # successor.status = Status.SCHEDULED
    #
    #         # if successor.unique_name in self.done_queue or successor.unique_name in self.running_queue:
    #         #     logger.debug(f"Task {successor.unique_name} " f"is currently running or already finished.")
    #         # else:
    #         #     logger.debug(f"Is now scheduling {successor.unique_name}.")
    #         #     self.scheduled_queue.put(successor.unique_name)
    #         #     # We have to add the successor id to the running queue here already
    #         #     # Assume, 2 workers execute a task each in parallel, finish at the same time
    #         #     # and both tasks have the same successor task.
    #         #     # Due to the lock only 1 worker enters this fn and puts his task's successor ids
    #         #     # in the scheduled queue. Once the worker leaves this fn the lock gets released
    #         #     # and the second worker adds his task's successors to the queue.
    #         #     # Now, if the shared successor id hasn't been picked up yet by another worker
    #         #     # which previously put the id in the running queue, the second worker puts
    #         #     # the successor id again in the schedule queue.
    #         #     # That leads to the task being executed more than once.
    #         #     # Hence, we have to add the successor ids to the running queue at the moment they are
    #         #     # added to the schedule queue.
    #         #     self.running_queue.append(successor.unique_name)


def run_task(task: Task) -> bool:

    # try to load existing run info object
    # if run info object was found, overwrite run info attributes with cached run info
    stored_run_info = task.load("fluidml_run_info")
    if stored_run_info:
        for k, v in stored_run_info.items():
            setattr(task, k, v)

    # provide task's result store with run info -> needed to properly name new run dirs
    task.results_store.run_info = task.info

    # if force is true, delete all task results and re-run task
    if task.force:
        with change_logging_level(level=50):
            task.delete_run()

    # check if task was successfully completed before
    completed: bool = task.results_store.is_finished(task_name=task.name, task_unique_config=task.unique_config)

    # if task is not completed, run the task now
    if not completed:
        # extract predecessor results
        controller = TaskDataController(task)
        pred_results: Dict = controller.pack_predecessor_results()

        # get run context from ResultStore
        run_context = task.get_store_context()
        if run_context:
            task.sweep_counter = run_context.sweep_counter

        # set and update the task's run history
        if not stored_run_info:
            task.run_history = {task.name: task.id}
            for pred in task.predecessors:
                pred_run_info = task.load(
                    "fluidml_run_info", task_name=pred.name, task_unique_config=pred.unique_config
                )
                task.run_history = {**pred_run_info["run_history"], **task.run_history}

        # save the run info object as json
        task.results_store.save(
            task.info.dict(),
            "fluidml_run_info",
            type_="json",
            task_name=task.name,
            task_unique_config=task.unique_config,
            indent=4,
        )

        if stored_run_info:
            logger.info(f'Started task "{task.unique_name}" with existing id "{task.id}"')
        else:
            logger.info(f'Started task "{task.unique_name}"')
        task.run(**pred_results)

        # save the "completed = 1" file
        task.results_store.save(
            "1",
            ".completed",
            type_="event",
            sub_dir=".load_info",
            task_name=task.name,
            task_unique_config=task.unique_config,
        )

    return completed
