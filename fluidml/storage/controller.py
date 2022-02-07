from collections import defaultdict
import inspect
from typing import Dict, Any, List, Optional

from fluidml.common import Task
from fluidml.common.exception import TaskResultKeyAlreadyExists, TaskResultObjectMissing
from fluidml.flow.task_spec import TaskSpec
from fluidml.storage import ResultsStore, Promise, Sweep, LazySweep


class TaskDataController:

    def __init__(self, task: Task):
        self._task_name: str = task.name
        self._results_store: ResultsStore = task.results_store
        self._predecessor_tasks: List[TaskSpec] = task.predecessors
        self._task_expects: Dict[str, inspect.Parameter] = task.expects
        self._reduce = task.reduce

    @staticmethod
    def _is_lazy(param: inspect.Parameter) -> bool:
        if param.annotation in [Promise, Optional[Promise], List[LazySweep]]:
            return True
        else:
            return False

    def _get_filtered_results_from_predecessor(self, predecessor: TaskSpec) -> Dict:
        if not predecessor.publishes:
            raise TaskResultObjectMissing(f'{self._task_name} expects {list(self._task_expects)} but predecessor '
                                          f'did not publish any results.')

        results = {}
        for item_name in predecessor.publishes:
            if item_name in self._task_expects:
                param: inspect.Parameter = self._task_expects[item_name]
                lazy: bool = self._is_lazy(param)
                obj: Optional[Any] = self._results_store.load(
                    name=item_name,
                    task_name=predecessor.name,
                    task_unique_config=predecessor.unique_config,
                    lazy=lazy
                )
                if obj is not None:
                    results[item_name] = obj
        return results

    def _get_results_from_predecessor(self, predecessor: TaskSpec) -> Dict:

        if self._task_expects:
            # get only expected results by the task from predecessor tasks
            results = self._get_filtered_results_from_predecessor(predecessor=predecessor)
        else:
            # get all published results from predecessor task
            results = self._results_store.get_results(task_name=predecessor.name,
                                                      task_unique_config=predecessor.unique_config,
                                                      task_publishes=predecessor.publishes)
        return results

    def pack_predecessor_results(self) -> Dict[str, Any]:
        if self._reduce:
            predecessor_results = defaultdict(list)
            for predecessor in self._predecessor_tasks:

                results = self._get_results_from_predecessor(predecessor=predecessor)

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

                for name, obj in results.items():
                    if name in predecessor_results.keys():
                        raise TaskResultKeyAlreadyExists(
                            f"{predecessor.name} saves a key '{name}' that already exists in another tasks' results")
                    else:
                        predecessor_results[name] = obj

        # Assertion to check that all expected results are retrieved
        if self._task_expects:
            retrieved_inputs = set(predecessor_results.keys())
            if retrieved_inputs != set(self._task_expects.keys()):
                missing_inputs = list(
                    set(self._task_expects).difference(retrieved_inputs))

                # remove args from missing inputs if a default value is registered in the task run signature
                missing_inputs = [arg for arg in missing_inputs
                                  if self._task_expects[arg].default is self._task_expects[arg].empty]
                if missing_inputs:
                    raise TaskResultObjectMissing(f'{self._task_name}: Result objects {missing_inputs} '
                                                  f'are required but could not be collected from predecessor tasks.')
        return predecessor_results


def pack_pipeline_results(all_tasks: List[TaskSpec],
                          results_store: ResultsStore,
                          return_results: bool = True) -> Dict[str, Any]:
    pipeline_results = defaultdict(list)
    if return_results:
        for task in all_tasks:
            results = results_store.get_results(task_name=task.name,
                                                task_unique_config=task.unique_config,
                                                task_publishes=task.publishes)
            pipeline_results[task.name].append({'results': results,
                                                'config': task.unique_config})
    else:
        for task in all_tasks:
            pipeline_results[task.name].append(task.unique_config)

    return _simplify_results(pipeline_results=pipeline_results)


def _simplify_results(pipeline_results: Dict[str, Any]) -> Dict[str, Any]:
    for task_name, task_results in pipeline_results.items():
        if len(task_results) == 1:
            pipeline_results[task_name] = task_results[0]
    return pipeline_results
