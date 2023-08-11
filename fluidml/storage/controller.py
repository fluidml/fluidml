import inspect
from collections import defaultdict
from typing import Any, Dict, List, Optional

from fluidml.exception import TaskResultKeyAlreadyExists, TaskResultObjectMissing
from fluidml.storage.base import LazySweep, Promise, ResultsStore, Sweep
from fluidml.task import Task, TaskResults
from fluidml.utils import change_logging_level


class TaskDataController:
    def __init__(self, task: Task):
        self._task_name: str = task.name
        self._results_store: ResultsStore = task.results_store
        self._predecessor_tasks: List[Task] = task.predecessors
        self._task_expects: Dict[str, inspect.Parameter] = task.expects
        self._reduce = task.reduce

    @staticmethod
    def _is_lazy(param: inspect.Parameter) -> bool:
        if param.annotation in [Promise, Optional[Promise], List[LazySweep]]:
            return True
        else:
            return False

    def _get_results_from_predecessor(self, predecessor: Task) -> Dict:
        results = {}
        for item_name in self._task_expects:
            param: inspect.Parameter = self._task_expects[item_name]
            lazy: bool = self._is_lazy(param)
            with change_logging_level(level=40):
                obj: Optional[Any] = self._results_store.load(
                    name=item_name,
                    task_name=predecessor.name,
                    task_unique_config=predecessor.unique_config,
                    lazy=lazy,
                )
            if obj is not None:
                results[item_name] = obj
        return results

    def pack_predecessor_results(self) -> Dict[str, Any]:
        if self._reduce:
            predecessor_results = defaultdict(list)
            for predecessor in self._predecessor_tasks:

                results = self._get_results_from_predecessor(predecessor=predecessor)

                if results is None:
                    raise TaskResultObjectMissing(
                        f"Some or all results from task {predecessor.name} could not be retrieved. "
                        f"Either the task did not publish any results, some published results are missing, "
                        f"or the task was not finished, yet."
                    )

                for name, obj in results.items():
                    if isinstance(obj, Promise):
                        sweep = LazySweep(value=obj, config=predecessor.unique_config)
                    else:
                        sweep = Sweep(value=obj, config=predecessor.unique_config)
                    predecessor_results[name].append(sweep)
        else:
            predecessor_results = {}
            for predecessor in self._predecessor_tasks:
                results = self._get_results_from_predecessor(predecessor=predecessor)

                if results is None:
                    raise TaskResultObjectMissing(
                        f"Some or all results from task {predecessor.name} could not be retrieved. "
                        f"Either the task did not publish any results, some published results are missing, "
                        f"or the task was not finished, yet."
                    )

                for name, obj in results.items():
                    if name in predecessor_results.keys():
                        raise TaskResultKeyAlreadyExists(
                            f"{predecessor.name} saves a key '{name}' that already exists in another tasks' results"
                        )
                    else:
                        predecessor_results[name] = obj

        # Assertion to check that all expected results are retrieved
        retrieved_inputs = set(predecessor_results.keys())
        if retrieved_inputs != set(self._task_expects.keys()):
            missing_inputs = list(set(self._task_expects).difference(retrieved_inputs))

            # remove args from missing inputs if a default value is registered in the task run signature
            missing_inputs = [
                arg for arg in missing_inputs if self._task_expects[arg].default is self._task_expects[arg].empty
            ]
            if missing_inputs:
                raise TaskResultObjectMissing(
                    f"{self._task_name}: Result objects {missing_inputs} "
                    f"are required but could not be collected from predecessor tasks."
                )
        return predecessor_results


def pack_pipeline_results(
    all_tasks: List[Task], return_results: Optional[str] = None
) -> Optional[Dict[str, List[TaskResults]]]:

    pipeline_results = defaultdict(list)
    if return_results is None:
        return None

    elif return_results == "all":
        for task in all_tasks:
            results = _get_saved_task_results(task=task)
            pipeline_results[task.name].append(results)

    elif return_results == "latest":
        for task in all_tasks:
            if not task.successors:
                results = _get_saved_task_results(task=task)
                pipeline_results[task.name].append(results)
    else:
        choices = ", ".join(['"all"', '"latest"', "None"])
        raise ValueError(f"return_results must be set to one of: {choices}.")

    return pipeline_results


def _get_saved_task_results(task: Task) -> TaskResults:
    results = task.results_store.get_results(
        task_name=task.name,
        task_unique_config=task.unique_config,
    )

    return TaskResults(
        task_name=task.name,
        task_unique_name=task.unique_name,
        results=results,
        unique_config=task.unique_config,
    )
