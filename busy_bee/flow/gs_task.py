from collections import MutableMapping
from copy import deepcopy
from itertools import product
from typing import Dict, Any, List, Tuple

from busy_bee.hive.task import Task


class GridSearch(Task):
    def __init__(self, task: type, name: str, gs_config: Dict[str, Any]):
        """
        A wrapper class for making tasks to be grid searcheable.

        Args:
            name (str): [description]
            task (Task): [description]
            gs_config (Dict[str, Any]): [description]
        """
        super().__init__(name=name)
        self.wrapped_task = task
        self.task_configs = self._split_gs_config(config_grid_search=gs_config)

    def run(self, results, resource):
        raise ValueError('This Metatask cannot be run.')

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
