from collections import defaultdict
from typing import Dict, Any, List, Optional

from fluidml.common import Task
from fluidml.common.exception import TaskResultKeyAlreadyExists, TaskResultObjectMissing
from fluidml.storage import ResultsStore


def pack_results(all_tasks: List[Task],
                 results_store: ResultsStore,
                 return_results: bool = True) -> Dict[str, Any]:
    results = defaultdict(list)
    if return_results:
        for task in all_tasks:
            result = results_store.get_results(task_name=task.name,
                                               task_unique_config=task.unique_config,
                                               task_publishes=task.publishes)
            results[task.name].append({'result': result,
                                       'config': task.unique_config})
    else:
        for task in all_tasks:
            results[task.name].append(task.unique_config)

    return simplify_results(results=results)


def get_filtered_results_from_predecessor(predecessor: Task,
                                          task_expects: List[str],
                                          results_store: ResultsStore) -> Dict:
    result = {}
    for item_name in predecessor.publishes:
        if item_name in task_expects:
            obj: Optional[Any] = results_store.load(name=item_name,
                                                    task_name=predecessor.name,
                                                    task_unique_config=predecessor.unique_config)
            if obj is not None:
                result[item_name] = obj
    return result


def get_results_from_predecessor(predecessor: Task,
                                 task_expects: List[str],
                                 results_store: ResultsStore) -> Dict:
    if task_expects is None:
        # get all published results from predecessor task
        result = results_store.get_results(task_name=predecessor.name,
                                           task_unique_config=predecessor.unique_config,
                                           task_publishes=predecessor.publishes)
    else:
        # get only expected results by the task from predecessor tasks
        result = get_filtered_results_from_predecessor(predecessor=predecessor,
                                                       task_expects=task_expects,
                                                       results_store=results_store)
    return result


def pack_predecessor_results(predecessor_tasks: List[Task],
                             results_store: ResultsStore,
                             reduce_task: bool,
                             task_expects: Optional[List[str]] = None) -> Dict[str, Any]:
    if reduce_task:
        all_results = []
        for predecessor in predecessor_tasks:

            result = get_results_from_predecessor(predecessor=predecessor,
                                                  task_expects=task_expects,
                                                  results_store=results_store)

            all_results.append({'result': result,
                                'config': predecessor.unique_config})

        # Assertion to check that all expected results are retrieved
        if task_expects is not None:
            retrieved_inputs = {
                name for result in all_results for name in result['result'].keys()}
            if retrieved_inputs != set(task_expects):
                missing_input_results = list(
                    set(task_expects).difference(retrieved_inputs))
                raise TaskResultObjectMissing(f'Result objects {missing_input_results} are required '
                                              f'but could not be collected from predecessor tasks.')
        return {"reduced_results": all_results}

    else:
        results = {}
        for predecessor in predecessor_tasks:
            result = get_results_from_predecessor(predecessor=predecessor,
                                                  task_expects=task_expects,
                                                  results_store=results_store)

            for key, value in result.items():
                if key in results.keys():
                    raise TaskResultKeyAlreadyExists(
                        f"{predecessor.name} saves a key '{key}' that already exists in another tasks's result")
                else:
                    results[key] = value

        # Assertion to check that all expected results are retrieved
        if task_expects is not None:
            retrieved_inputs = {name for name in results.keys()}
            if retrieved_inputs != set(task_expects):
                missing_input_results = list(
                    set(task_expects).difference(retrieved_inputs))
                raise TaskResultObjectMissing(f'Result objects {missing_input_results} are required '
                                              f'but could not be collected from predecessor tasks.')
    return results


def simplify_results(results: Dict[str, Any]) -> Dict[str, Any]:
    for task_name, task_results in results.items():
        if len(task_results) == 1:
            results[task_name] = task_results[0]
    return results
