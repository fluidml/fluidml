import datetime
import json
import logging
import sys
from abc import abstractmethod
from multiprocessing import Event, Process, Queue, RLock
from queue import Empty
from typing import Any, Dict, Optional, Union

from fluidml.logging import QueueHandler, StderrHandler, StdoutHandler
from fluidml.storage.base import Names
from fluidml.storage.controller import TaskDataController
from fluidml.task import Task, TaskInfo, TaskState
from fluidml.utils import change_logging_level

logger = logging.getLogger(__name__)


class Whale(Process):
    def __init__(
        self,
        exit_event: Event,
        exit_on_error: bool,
        logging_queue: Queue,
        error_queue: Queue,
        logging_lvl: int,
        lock: RLock,
        use_multiprocessing: bool = True,
    ):
        self.use_multiprocessing = use_multiprocessing
        # only init the Process parent class if multiprocessing is configured.
        if self.use_multiprocessing:
            super().__init__(target=self.work, args=())
            self._logging_queue = logging_queue
            self._logging_lvl = logging_lvl

        self.exit_event = exit_event
        self._exit_on_error = exit_on_error
        self._error_queue = error_queue
        self._lock = lock

    def _configure_logging(self):
        h = QueueHandler(self._logging_queue, self.name)
        root = logging.getLogger()
        root.addHandler(h)
        root.setLevel(self._logging_lvl)

    def _redirect_stdout_stderr(self):
        sys.stdout = StdoutHandler(self._logging_queue, self.name)
        sys.stderr = StderrHandler(self._logging_queue, self.name)

    @abstractmethod
    def _work(self):
        raise NotImplementedError

    def work(self):
        try:
            if self.use_multiprocessing:
                self._redirect_stdout_stderr()
                self._configure_logging()
            self._work()
        except Exception as e:
            logger.exception(e)
            self._error_queue.put(e)
            self.exit_event.set()


class Dolphin(Whale):
    def __init__(
        self,
        scheduled_queue: Queue,
        task_states: Dict[str, TaskState],
        error_queue: Queue,
        lock: RLock,
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
            if self.task_states[predecessor.unique_name] in [
                TaskState.FAILED,
                TaskState.UPSTREAM_FAILED,
            ]:
                return True
        return False

    def _is_task_ready(self, task: Task):
        for predecessor in task.predecessors:
            if self.task_states[predecessor.unique_name] != TaskState.FINISHED:
                return False
        return True

    def _fetch_next_task(self) -> Union[str, None]:
        try:
            task_counter = self.scheduled_queue.get(block=True, timeout=0.05)
        except Empty:
            task_counter = None
        return task_counter

    def _schedule_successors(self, task: Task):
        # get successor tasks and put them in task queue for processing
        for successor in task.successors:
            if self.task_states[successor.unique_name] == TaskState.PENDING:
                logger.debug(f"Is now scheduling {successor.unique_name}.")
                self.task_states[successor.unique_name] = TaskState.SCHEDULED
                self.scheduled_queue.put(successor.unique_name)

    def _log_task_finished(self, task: Task, completed: bool):
        if completed:
            msg = f'Task "{task.unique_name}" already executed'
        else:
            msg = f'Finished task "{task.unique_name}"'

        num_finished_task = sum(1 for t in self.task_states.values() if t == TaskState.FINISHED)
        logger.info(
            f"{msg} [{num_finished_task}/{self.num_tasks} " f"- {round((num_finished_task / self.num_tasks) * 100)}%]"
        )

    def _on_task_start(self, task: Task, stored_task_info: Optional[Dict] = None):
        """Prepare task starting.

        Logging, update the task state, set the start time, call the user's task's __init__ function.
        """

        if stored_task_info:
            logger.info(f'Started task "{task.unique_name}" with existing id "{task.id}"')
        else:
            logger.info(f'Started task "{task.unique_name}"')

        # update state and register started timestamp
        self.task_states[task.unique_name] = TaskState.RUNNING
        task.state = TaskState.RUNNING
        task.started = datetime.datetime.now()

        # call user defined __init__ of task object
        task._init()

    def _on_task_end(
        self,
        task: Task,
        completed: Optional[bool] = None,
        exception: Optional[BaseException] = None,
    ):
        logger.debug(f'Enter "_on_task_end()" with status "{self.task_states[task.unique_name]}"')

        # register ended timestamp
        task.ended = datetime.datetime.now()

        if self.task_states[task.unique_name] == TaskState.FINISHED:
            # Log task completion
            self._log_task_finished(task, completed)

        elif self.task_states[task.unique_name] == TaskState.KILLED:
            # Log exception and put it in error queue
            logger.exception(f'Killed task "{task.unique_name}"\n{exception}')
            self._error_queue.put(exception)
            # set exit event to stop process
            self.exit_event.set()

        elif self.task_states[task.unique_name] == TaskState.FAILED:
            # Log exception and put it in error queue
            logger.exception(f'Task "{task.unique_name}" failed with error:\n{exception}')
            self._error_queue.put(exception)
            # set exit event to stop process only if exit on error is set to True
            if self._exit_on_error:
                self.exit_event.set()

        # save the fluidml info object as json
        task.state = self.task_states[task.unique_name]
        # we use pydantics 2.x model_dump_json() fn, load the json str as dict and save it
        try:
            task_info_json = task.info.model_dump_json()
        except AttributeError:
            # Fallback to pydantic 1.x .json() function
            task_info_json = task.info.json()
        task.results_store.save(
            json.loads(task_info_json),
            Names.FLUIDML_INFO_FILE,
            type_="json",
            task_name=task.name,
            task_unique_config=task.unique_config,
            indent=4,
        )

    def _execute_task(self, task: Task):
        # provide resource and lock
        task.resource = self.resource
        task.results_store.lock = self._lock

        # try to load existing task info object
        # if task info object was found, overwrite run info attributes with cached run info
        with change_logging_level(40):
            stored_task_info = task.load(Names.FLUIDML_INFO_FILE)
        if stored_task_info:
            # assign loaded objects to current task
            for k, v in TaskInfo(**stored_task_info).dict().items():
                # try/except block for property attributes (id, duration) which cannot be set
                try:
                    setattr(task, k, v)
                except AttributeError:
                    pass

            # update tasks state to stored state
            self.task_states[task.unique_name] = task.state  # stored_task_info.state

        # provide task's result store with run info -> needed to properly name new run dirs
        task.results_store.run_info = task.info

        # if force is true, delete all task results and re-run task
        if task.force:
            with change_logging_level(level=40):
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
            if not stored_task_info:
                task.run_history = {task.name: task.id}
                for pred in task.predecessors:
                    pred_run_info = task.load(
                        Names.FLUIDML_INFO_FILE,
                        task_name=pred.name,
                        task_unique_config=pred.unique_config,
                    )
                    task.run_history = {
                        **pred_run_info["run_history"],
                        **task.run_history,
                    }

            try:
                # start the task
                self._on_task_start(task=task, stored_task_info=stored_task_info)

                # run the task
                task.run(**pred_results)

                # finish the task
                with self._lock:
                    self.task_states[task.unique_name] = TaskState.FINISHED
                    self._on_task_end(task)

            except KeyboardInterrupt as e:
                # handle KeyboardInterrupt
                self.task_states[task.unique_name] = TaskState.KILLED
                self._on_task_end(task, exception=e)

            except Exception as e:
                # handle other exceptions
                self.task_states[task.unique_name] = TaskState.FAILED
                self._on_task_end(task, exception=e)

        with self._lock:
            if completed:
                self._log_task_finished(task, completed)
            self._schedule_successors(task)

    def _stop(self) -> bool:
        return self.exit_event.is_set() or all(
            True if t in [TaskState.FINISHED, TaskState.FAILED, TaskState.UPSTREAM_FAILED] else False
            for t in self.task_states.values()
        )

    def _work(self):
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
