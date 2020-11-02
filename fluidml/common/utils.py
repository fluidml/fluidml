from typing import Any, Dict, Callable, Union, Tuple, Optional

from fluidml.common import Task, Resource


class MyTask(Task):
    """A constructor class that creates a task object from a callable."""

    def __init__(self,
                 name: str,
                 id_: int,
                 task: Callable,
                 kwargs: Dict,
                 force: Optional[bool] = None):
        super().__init__(id_=id_, name=name)
        self.task = task
        self.kwargs = kwargs
        self.force = force

    def run(self, results: Dict[str, Any], resource: Resource):
        result = self.task(results, resource, **self.kwargs)
        return result


def update_merge(d1: Dict, d2: Dict) -> Union[Dict, Tuple]:
    if isinstance(d1, dict) and isinstance(d2, dict):
        # Unwrap d1 and d2 in new dictionary to keep non-shared keys with **d1, **d2
        # Next unwrap a dict that treats shared keys
        # If two keys have an equal value, we take that value as new value
        # If the values are not equal, we recursively merge them
        return {**d1, **d2,
                **{k: d1[k] if d1[k] == d2[k] else update_merge(d1[k], d2[k])
                    for k in {*d1} & {*d2}}
                }
    else:
        # This case happens when values are merged
        # It bundle values in a tuple, assuming the original dicts
        # don't have tuples as values
        if isinstance(d1, tuple) and not isinstance(d2, tuple):
            combined = d1 + (d2,)
        elif isinstance(d2, tuple) and not isinstance(d1, tuple):
            combined = d2 + (d1,)
        elif isinstance(d1, tuple) and isinstance(d2, tuple):
            combined = d1 + d2
        else:
            combined = (d1, d2)
        return tuple(sorted(element for i, element in enumerate(combined) if element not in combined[:i]))


def reformat_config(d: Dict) -> Dict:
    for key, value in d.items():
        if isinstance(value, list):
            d[key] = [value]
        if isinstance(value, tuple):
            d[key] = list(value)
        elif isinstance(value, dict):
            reformat_config(value)
        else:
            continue
    return d