from typing import Dict, Any, Optional, List, Tuple
from busy_bee.common.dependency import DependencyMixin
from collections import MutableMapping
from copy import deepcopy
from itertools import product


class TaskSpec(DependencyMixin):
    def __init__(self, task_cls: type, task_kwargs: Dict[str, Any], name: Optional[str] = None):
        """
        A class to hold task specification

        Args:
            task_cls (type): a task class to instantiate
            task_kwargs (Dict[str, Any]): task arguments used while instantiating
            name (Optional[str], optional): [description]. Defaults to None
        """
        self._task_cls = task_cls
        self._task_kwargs = task_kwargs
        self._name = name if name is not None else self._task_cls.__name__


class GridTaskSpec(DependencyMixin):
    def __init__(self, task_cls: type, gs_config: Dict[str, Any], name: Optional[str] = None):
        """
        A class to hold task specification for grid searcheable task

        Args:
            task_cls (type): a task class to instantiate and expand
            gs_config (Dict[str, Any]): a grid search config that will be expanded
            name (Optional[str], optional): [description]. Defaults to None
        """
        self._task_cls = task_cls
        self._gs_config = gs_config
        self._name = name if name is not None else self._task_cls.__name__

    def _find_list_in_dict(self, obj: Dict, param_grid: List) -> List:
        for key in obj:
            if isinstance(obj[key], list):
                param_grid.append([val for val in obj[key]])
            elif isinstance(obj[key], dict):
                self._find_list_in_dict(obj[key], param_grid)
            else:
                continue
        return param_grid

    def _replace_list_in_dict(self, obj: Dict, obj_copy: Dict, comb: Tuple, counter: List) -> Tuple[Dict, List]:
        for key, key_copy in zip(obj, obj_copy):
            if isinstance(obj[key], list):
                obj_copy[key_copy] = comb[len(counter)]
                counter.append(1)
            elif isinstance(obj[key], dict):
                self._replace_list_in_dict(obj[key], obj_copy[key_copy], comb, counter)
            else:
                continue
        return obj_copy, counter

    def _dict_to_list_of_strings(self, d: Dict, parent_key: str = '', sep: str = '.'):
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, MutableMapping):
                items.extend(self._dict_to_list_of_strings(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                items.append((new_key, str(v)))
            else:
                items.append((new_key, v))
        return [str(k + sep + str(v)) for k, v in items]

    def _split_gs_config(self, config_grid_search: Dict) -> List[Dict]:
        param_grid = []
        param_grid = self._find_list_in_dict(config_grid_search, param_grid)
        config_copy = deepcopy(config_grid_search)
        individual_configs = []
        for comb in product(*param_grid):
            counter = []
            individual_config = self._replace_list_in_dict(config_grid_search, config_copy, comb, counter)[0]
            individual_config = deepcopy(individual_config)
            individual_configs.append(individual_config)
        return individual_configs
