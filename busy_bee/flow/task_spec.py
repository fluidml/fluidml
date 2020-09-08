from copy import deepcopy
from itertools import product
from types import FunctionType
from typing import Dict, Any, Optional, List, Tuple, Union

from busy_bee.common.dependency import DependencyMixin


class TaskSpec(DependencyMixin):
    """
    A class to hold task specification

    Args:
        task (type): a task class to instantiate
        task_kwargs (Dict[str, Any]): task arguments used while instantiating
        name (Optional[str], optional): Defaults to None
    """

    def __init__(self,
                 task: Union[type, FunctionType],
                 task_kwargs: Dict[str, Any],
                 name: Optional[str] = None):
        super().__init__()

        self.task = task
        self.task_kwargs = task_kwargs
        self.name = name if name is not None else self.task.__name__


class GridTaskSpec(DependencyMixin):
    """
    A class to hold task specification for grid searcheable task

    Args:
        task (type): a task class to instantiate and expand
        gs_config (Dict[str, Any]): a grid search config that will be expanded
        name (Optional[str], optional): Defaults to None
    """

    def __init__(self,
                 task: Union[type, FunctionType],
                 gs_config: Dict[str, Any],
                 name: Optional[str] = None):
        super().__init__()

        self.task = task
        # self.gs_config = gs_config
        self.task_configs = self._split_gs_config(config_grid_search=gs_config)
        self.name = name if name is not None else self.task.__name__

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
