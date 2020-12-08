from collections import defaultdict
from typing import Dict, Any, List, Tuple

from fluidml.common.exception import TaskResultTypeError
from fluidml.storage import ResultsStore


def pack_results(results_store: ResultsStore,
                 task_configs: List[Tuple[str, Dict]],
                 return_results: bool = True) -> Dict[str, Any]:
    results = defaultdict(list)
    if return_results:
        for task_name, task_config in task_configs:
            result = results_store.get_results(task_name, task_config)
            if isinstance(result, dict):
                results[task_name].append({'result': result,
                                           'config': task_config})
            else:
                raise TaskResultTypeError("Each task has to return a dict")
    else:
        for task_name, task_config in task_configs:
            results[task_name].append(task_config)

    return simplify_results(results=results)


def simplify_results(results: Dict[str, Any]) -> Dict[str, Any]:
    for task_name, task_results in results.items():
        if len(task_results) == 1:
            results[task_name] = task_results[0]
    return results
