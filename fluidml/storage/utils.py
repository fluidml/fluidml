from typing import Dict, Any, List, Tuple
from fluidml.storage.base import ResultsStore
from collections import defaultdict
from fluidml.common.exception import TaskResultTypeError


def pack_results(results_store: ResultsStore, task_configs: List[Tuple[str, Dict]]) -> Dict[str, Any]:
    results = defaultdict(list)
    for task_name, task_config in task_configs:
        result = results_store.get_results(task_name, task_config)
        if isinstance(result, dict):
            results[task_name].append(result)
        else:
            raise TaskResultTypeError("Each task has to return a dict")

    # simplify results
    for task_name, task_results in results.items():
        if len(task_results) == 1:
            results[task_name] = task_results[0]

    return results
