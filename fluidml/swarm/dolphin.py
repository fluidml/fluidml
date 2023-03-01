import logging
import time
import signal
from multiprocessing import Queue, Lock, Event
from queue import Empty
from typing import Dict, Any, Union, Optional

from fluidml.common import Task
from fluidml.common.utils import change_logging_level
from fluidml.storage.controller import TaskDataController
from fluidml.swarm import Whale

from fluidml.common.task import TaskState

logger = logging.getLogger(__name__)


class Dolphin(Whale):
    def __init__(
        self,
        scheduled_queue: Queue,
        task_states: Dict[str, TaskState],
        error_queue: Queue,
        lock: Lock,
        tasks: Dict[str, Task],
        exit_event: Event,
        exit_on_error: bool,
        logging_lvl: int,
        logging_queue: Optional[Queue] = None,
        resource: Optional[Any] = None,
        use_multiprocessing: bool = True,
    ):
        super().__init__(
            exit_event=exit_event,
            exit_on_error=exit_on_error,
            logging_queue=logging_queue,
            error_queue=error_queue,
            logging_lvl=logging_lvl,
            lock=lock,
            use_multiprocessing=use_multiprocessing,
        )
        self.resource = resource
        self.scheduled_queue = scheduled_queue
        self.task_states = task_states
        self.tasks = tasks

    @property
    def num_tasks(self):
        return len(self.tasks)

    def _detect_upstream_error(self, task: Task) -> bool:
        for predecessor in task.predecessors:
            if self.task_states[predecessor.unique_name] in [TaskState.FAILED, TaskState.UPSTREAM_FAILED]:
                return True
        return False

    def _is_task_ready(self, task: Task):
        for predecessor in task.predecessors:
            if self.task_states[predecessor.unique_name] != TaskState.FINISHED:
                return False
        return True

    def _fetch_next_task(self) -> Union[str, None]:
        try:
            task_counter = self.scheduled_queue.get(block=False, timeout=0.5)
        except Empty:
            task_counter = None
        return task_counter

    # def _execute_task(self, task: Task):
    #
    #     # run the task
    #     completed: bool = self._run_task(task)
    #
    #     with self._lock:
    #         self.task_states[task.unique_name] = TaskState.FINISHED
    #
    #         # Log task completion
    #         if completed:
    #             msg = f'Task "{task.unique_name}" already executed'
    #         else:
    #             msg = f'Finished task "{task.unique_name}"'
    #
    #         num_finished_task = sum(1 for t in self.task_states.values() if t == TaskState.FINISHED)
    #         logger.info(
    #             f"{msg} [{num_finished_task}/{self.num_tasks} "
    #             f"- {round((num_finished_task / self.num_tasks) * 100)}%]"
    #         )

    def _execute_task(self, task: Task):

        # provide resource and lock
        task.resource = self.resource
        task.results_store.lock = self._lock

        # try to load existing run info object
        # if run info object was found, overwrite run info attributes with cached run info
        stored_run_info = task.load("fluidml_run_info")
        if stored_run_info:
            # assign loaded objects to current task
            for k, v in stored_run_info.items():
                setattr(task, k, v)
            self.task_states[task.unique_name] = stored_run_info["state"]

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

            # # save the run info object as json
            # task.results_store.save(
            #     task.info.dict(),
            #     "fluidml_run_info",
            #     type_="json",
            #     task_name=task.name,
            #     task_unique_config=task.unique_config,
            #     indent=4,
            # )

            if stored_run_info:
                logger.info(f'Started task "{task.unique_name}" with existing id "{task.id}"')
            else:
                logger.info(f'Started task "{task.unique_name}"')

            # instantiate user defined __init__ of task object
            self.task_states[task.unique_name] = TaskState.RUNNING
            task.state = TaskState.RUNNING
            try:
                task._init()
                task.run(**pred_results)

                # update task status
                with self._lock:
                    self.task_states[task.unique_name] = TaskState.FINISHED
                    msg = f'Finished task "{task.unique_name}"'

                    num_finished_task = sum(1 for t in self.task_states.values() if t == TaskState.FINISHED)
                    logger.info(
                        f"{msg} [{num_finished_task}/{self.num_tasks} "
                        f"- {round((num_finished_task / self.num_tasks) * 100)}%]"
                    )

            except KeyboardInterrupt as e:
                # log exception
                # logger.warning(f'Killed task "{task.unique_name}"')
                # e.args = (f'Killed task "{task.unique_name}"',)
                logger.exception(f'Killed task "{task.unique_name}"\n{e}')
                self._error_queue.put(e)
                # update task status
                self.task_states[task.unique_name] = TaskState.KILLED
                # set exit event to stop process
                self.exit_event.set()
            except Exception as e:
                # logger.warning(f"Task {task.unique_name} failed with error:")
                logger.exception(f'Task "{task.unique_name}" failed with error:\n{e}')
                self._error_queue.put(e)
                self.task_states[task.unique_name] = TaskState.FAILED
                if self._exit_on_error:
                    self.exit_event.set()

            # save the run info object as json
            task.state = self.task_states[task.unique_name]
            task.results_store.save(
                task.info.dict(),
                "fluidml_run_info",
                type_="json",
                task_name=task.name,
                task_unique_config=task.unique_config,
                indent=4,
            )

        with self._lock:
            if completed:
                # Log task completion
                msg = f'Task "{task.unique_name}" already executed'

                num_finished_task = sum(1 for t in self.task_states.values() if t == TaskState.FINISHED)
                logger.info(
                    f"{msg} [{num_finished_task}/{self.num_tasks} "
                    f"- {round((num_finished_task / self.num_tasks) * 100)}%]"
                )

            self._schedule_successors(task)

    def _stop(self) -> bool:

        return self.exit_event.is_set() or all(
            True if t in [TaskState.FINISHED, TaskState.FAILED, TaskState.UPSTREAM_FAILED] else False
            for t in self.task_states.values()
        )

    def _work(self):
        # signal.signal(signal.SIGINT, signal.SIG_IGN)
        while not self._stop():
            # fetch next task to run
            task_unique_name = self._fetch_next_task()

            # continue when there is a valid task to run
            if task_unique_name is not None:

                # get current task from task_unique_name
                task = self.tasks[task_unique_name]

                if self._detect_upstream_error(task):
                    self.task_states[task.unique_name] = TaskState.UPSTREAM_FAILED
                    self._schedule_successors(task)
                    logger.warning(f'Task "{task.unique_name}" cannot be executed due to an upstream task failure.')
                    continue

                # run task only if all dependencies are satisfied
                if not self._is_task_ready(task=task):
                    logger.debug(f"Dependencies are not satisfied yet for task {task.unique_name}")
                    self.scheduled_queue.put(task.unique_name)
                    continue

                # execute the task
                self._execute_task(task)

    def _schedule_successors(self, task: Task):
        # get successor tasks and put them in task queue for processing
        for successor in task.successors:
            if self.task_states[successor.unique_name] == TaskState.PENDING:
                logger.debug(f"Is now scheduling {successor.unique_name}.")
                self.task_states[successor.unique_name] = TaskState.SCHEDULED
                self.scheduled_queue.put(successor.unique_name)

    # def _work(self):
    #
    #     while not self._is_done():
    #
    #         # fetch next task to run
    #         task_unique_name = self._fetch_next_task()
    #
    #         # continue when there is a valid task to run
    #         if task_unique_name is not None:
    #             # get current task from task_unique_name
    #             task = self.tasks[task_unique_name]
    #
    #             try:
    #                 with self._lock:
    #                     if not self._is_task_ready(task):
    #                         continue
    #                     # # check if any predecessor task is in failed_tasks list
    #                     # failed_predecessor = None
    #                     # for predecessor in task.predecessors:
    #                     #     if self.task_states[predecessor.unique_name] in [
    #                     #         TaskState.FAILED,
    #                     #         TaskState.UPSTREAM_FAILED,
    #                     #     ]:
    #                     #         # if predecessor.state in [
    #                     #         #     TaskState.FAILED,
    #                     #         #     TaskState.UPSTREAM_FAILED,
    #                     #         # ]:
    #                     #         failed_predecessor = predecessor.unique_name
    #                     #         break
    #                     #
    #                     # if failed_predecessor:
    #                     #     logger.info(
    #                     #         f'Task "{task_unique_name}" cannot be executed since predecessor task "{failed_predecessor}" failed.'
    #                     #     )
    #                     #     self.task_states[task.unique_name] = TaskState.UPSTREAM_FAILED
    #                     #     # task.state = TaskState.UPSTREAM_FAILED
    #                     #
    #                     #     self._schedule_successors(task)
    #                     #     continue
    #                     #
    #                     # # run task only if all dependencies are satisfied
    #                     # if not self._is_task_ready(task=task):
    #                     #     logger.debug(f"Dependencies are not satisfied yet for " f"task {task.unique_name}")
    #                     #     self.scheduled_queue.put(task.unique_name)
    #                     #     continue
    #
    #                 # instantiate user defined __init__ of task object
    #                 task._init()
    #
    #                 self.task_states[task.unique_name] = TaskState.RUNNING
    #                 # task.state = self.task_states[task.unique_name]
    #                 # task.state = TaskState.RUNNING
    #
    #                 # execute the task
    #                 self._execute_task(task)
    #
    #                 with self._lock:
    #                     # schedule the task's successors
    #                     self._schedule_successors(task)
    #             except Exception as e:
    #                 with self._lock:
    #                     logger.info(f'Task "{task_unique_name}" cannot be executed.')
    #                     self.task_states[task_unique_name] = TaskState.FAILED
    #                     # task.state = TaskState.FAILED
    #                     self._schedule_successors(task)
    #                     raise
    #
    # def _schedule_successors(self, task: Task):
    #     # get successor tasks and put them in task queue for processing
    #     for successor in task.successors:
    #         if self.task_states[successor.unique_name] == TaskState.PENDING:
    #             # if successor.state == TaskState.PENDING:
    #             logger.debug(f"Is now scheduling {successor.unique_name}.")
    #             self.scheduled_queue.put(successor.unique_name)
    #             self.task_states[successor.unique_name] = TaskState.SCHEDULED
    #             # successor.state = TaskState.SCHEDULED


# def run_task(task: Task) -> bool:
#
#     # try to load existing run info object
#     # if run info object was found, overwrite run info attributes with cached run info
#     stored_run_info = task.load("fluidml_run_info")
#     if stored_run_info:
#         for k, v in stored_run_info.items():
#             setattr(task, k, v)
#
#     # provide task's result store with run info -> needed to properly name new run dirs
#     task.results_store.run_info = task.info
#
#     # if force is true, delete all task results and re-run task
#     if task.force:
#         with change_logging_level(level=50):
#             task.delete_run()
#
#     # check if task was successfully completed before
#     completed: bool = task.results_store.is_finished(task_name=task.name, task_unique_config=task.unique_config)
#
#     # if task is not completed, run the task now
#     if not completed:
#         # extract predecessor results
#         controller = TaskDataController(task)
#         pred_results: Dict = controller.pack_predecessor_results()
#
#         # get run context from ResultStore
#         run_context = task.get_store_context()
#         if run_context:
#             task.sweep_counter = run_context.sweep_counter
#
#         # set and update the task's run history
#         if not stored_run_info:
#             task.run_history = {task.name: task.id}
#             for pred in task.predecessors:
#                 pred_run_info = task.load(
#                     "fluidml_run_info", task_name=pred.name, task_unique_config=pred.unique_config
#                 )
#                 task.run_history = {**pred_run_info["run_history"], **task.run_history}
#
#         # save the run info object as json
#         task.results_store.save(
#             task.info.dict(),
#             "fluidml_run_info",
#             type_="json",
#             task_name=task.name,
#             task_unique_config=task.unique_config,
#             indent=4,
#         )
#
#         if stored_run_info:
#             logger.info(f'Started task "{task.unique_name}" with existing id "{task.id}"')
#         else:
#             logger.info(f'Started task "{task.unique_name}"')
#
#         # instantiate user defined __init__ of task object
#         task._init()
#         task.run(**pred_results)
#
#         # save the "completed = 1" file
#         task.results_store.save(
#             "1",
#             ".completed",
#             type_="event",
#             sub_dir=".load_info",
#             task_name=task.name,
#             task_unique_config=task.unique_config,
#         )
#
#     return completed
