from copy import deepcopy
from itertools import product
from typing import Dict, Any, Optional, List, Tuple, Union, Callable

from fluidml.flow import BaseTaskSpec
from fluidml.common import Task


class TaskSpec(BaseTaskSpec):
    """A class to hold specification of a plain task

    Args:
        task (type): a task class to instantiate
        task_kwargs (Dict[str, Any]): task arguments used while instantiating
        name (Optional[str], optional): Defaults to None
    """

    def __init__(self,
                 task: Union[type, Callable],
                 task_kwargs: Optional[Dict[str, Any]] = {},
                 name: Optional[str] = None):
        super().__init__(task, name)
        self.task_kwargs = task_kwargs

    def build(self) -> List[Task]:
        task = self._create_task_object(task_id=None, task_kwargs=self.task_kwargs)
        return [task]


class GridTaskSpec(BaseTaskSpec):
    """A class to hold specification of a grid searcheable task

    Args:
        task (type): a task class to instantiate and expand
        gs_config (Dict[str, Any]): a grid search config that will be expanded
        name (Optional[str], optional): Defaults to None
    """

    def __init__(self,
                 task: Union[type, Callable],
                 gs_config: Dict[str, Any],
                 name: Optional[str] = None):
        super().__init__(task, name)
        self.task_configs = self._split_gs_config(config_grid_search=gs_config)

    def build(self) -> List[Task]:
        tasks = [self._create_task_object(task_id=None, task_kwargs=config) for config in self.task_configs]
        return tasks

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
