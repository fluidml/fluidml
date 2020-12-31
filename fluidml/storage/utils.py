from collections import defaultdict
from typing import Dict, Any, List, Tuple

from fluidml.common.exception import TaskResultTypeError, TaskResultKeyAlreadyExists
from fluidml.storage import ResultsStore


def pack_results(results_store: ResultsStore,
                 task_configs: List[Tuple[str, Dict]],
                 return_results: bool = True) -> Dict[str, Any]:
    results = defaultdict(list)
    if return_results:
        for task_name, task_config in task_configs:
            result = get_task_result(results_store, task_name, task_config)
            results[task_name].append({'result': result,
                                       'config': task_config})
    else:
        for task_name, task_config in task_configs:
            results[task_name].append(task_config)

    return simplify_results(results=results)


def get_task_result(results_store: ResultsStore,
                    task_name: str,
                    task_config: Dict[str, Any]) -> Dict[str, Any]:
    result = results_store.get_results(task_name, task_config)
    if isinstance(result, dict):
        return result
    else:
        raise TaskResultTypeError("Each task has to return a dict")


def pack_predecessor_results(results_store: ResultsStore,
                             task_configs: List[Tuple[str, Dict]],
                             reduce_task: bool) -> Dict[str, Any]:
    if reduce_task:
        all_results = []
        for task_name, task_config in task_configs:
            result = get_task_result(results_store, task_name, task_config)
            all_results.append({'result': result,
                                'config': task_config})
        return {"reduced_results": all_results}

    else:
        results = {}
        for task_name, task_config in task_configs:
            result = get_task_result(results_store, task_name, task_config)
            for key, value in result.items():
                if key in results.keys():
                    raise TaskResultKeyAlreadyExists(
                        f"{task_name}'s result dict has a key that already exists in another tasks's result")
                else:
                    results[key] = value
    return results


def simplify_results(results: Dict[str, Any]) -> Dict[str, Any]:
    for task_name, task_results in results.items():
        if len(task_results) == 1:
            results[task_name] = task_results[0]
    return results
