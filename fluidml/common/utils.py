from typing import Any, Dict, Callable, Union, Tuple

from metadict import MetaDict

from fluidml.common.task import Task


class MyTask(Task):
    """A constructor class that creates a task object from a callable."""

    def __init__(self, task: Callable, config_kwargs: MetaDict, additional_kwargs: MetaDict):
        super().__init__()
        self.task = task
        self.config_kwargs = config_kwargs
        self.additional_kwargs = additional_kwargs

    def run(self, **results: Dict[str, Any]):
        self.task(**results, **self.config_kwargs, **self.additional_kwargs, task=self)


def update_merge(d1: Dict, d2: Dict) -> Union[Dict, Tuple]:
    if isinstance(d1, dict) and isinstance(d2, dict):
        # Unwrap d1 and d2 in new dictionary to keep non-shared keys with **d1, **d2
        # Next unwrap a dict that treats shared keys
        # If two keys have an equal value, we take that value as new value
        # If the values are not equal, we recursively merge them
        return {**d1, **d2, **{k: d1[k] if d1[k] == d2[k] else update_merge(d1[k], d2[k]) for k in {*d1} & {*d2}}}
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


def remove_none_from_dict(obj: Dict) -> Dict:
    if isinstance(obj, (list, tuple, set)):
        return type(obj)(remove_none_from_dict(x) for x in obj if x is not None)
    elif isinstance(obj, dict):
        return {k: remove_none_from_dict(v) for k, v in obj.items() if k is not None and v is not None}
    else:
        return obj


def remove_prefixed_keys_from_dict(obj: Dict, prefix: str = "@") -> Dict:
    if isinstance(obj, (list, tuple, set)):
        return type(obj)(remove_prefixed_keys_from_dict(x, prefix) for x in obj)
    elif isinstance(obj, dict):
        return {
            k: remove_prefixed_keys_from_dict(v, prefix)
            for k, v in obj.items()
            if not isinstance(k, str) or (isinstance(k, str) and not k.startswith(prefix))
        }
    else:
        return obj


def remove_prefix_from_dict(obj: Dict, prefix: str = "@") -> Dict:
    if isinstance(obj, (list, tuple, set)):
        return type(obj)(remove_prefix_from_dict(x, prefix) for x in obj)
    elif isinstance(obj, dict):
        return {
            (k.split(prefix, 1)[-1] if isinstance(k, str) and k.startswith(prefix) else k): remove_prefix_from_dict(
                v, prefix
            )
            for k, v in obj.items()
        }
    else:
        return obj
