from abc import ABC, abstractmethod
from copy import deepcopy
from itertools import product
from typing import Dict, Any, Optional, List, Tuple, Union, Callable

from fluidml.common import Task, DependencyMixin
from fluidml.common.utils import MyTask
from fluidml.common.exception import TaskPublishesSpecMissing, GridSearchExpansionError


class BaseTaskSpec(DependencyMixin, ABC):
    def __init__(self,
                 task: Union[type, Callable],
                 name: Optional[str] = None,
                 reduce: Optional[bool] = None,
                 publishes: Optional[List[str]] = None,
                 expects: Optional[List[str]] = None):
        DependencyMixin.__init__(self)
        self.task = task
        self.name = name if name is not None else self.task.__name__
        self.reduce = reduce
        self.publishes = publishes
        self.expects = expects

        self.force: Optional[str] = None

    def _create_task_object(self,
                            task_kwargs: Dict[str, Any]) -> Task:
        if isinstance(self.task, type):
            task = self.task(**task_kwargs)
            task.kwargs = task_kwargs

        elif isinstance(self.task, Callable):
            task = MyTask(task=self.task, kwargs=task_kwargs)
        else:
            raise TypeError(f'{self.task} needs to be a Class object (type="type") or a Callable, e.g. a function.'
                            f'But it is of type "{type(self.task)}".')
        task.name = self.name

        # override publishes from task spec
        task = self._override_publishes(task)

        # expects
        if self.expects is not None:
            task.expects = self.expects

        return task

    def _override_publishes(self, task: Task):
        if self.publishes is not None:
            task.publishes = self.publishes

        if task.publishes is None:
            raise TaskPublishesSpecMissing(
                f'{self.task} needs "publishes" specification either in task definition or task specification')
        return task

    @abstractmethod
    def build(self) -> List[Task]:
        """Builds task from the specification

        Returns:
            List[Task]: task objects that are created
        """

        raise NotImplementedError


class TaskSpec(BaseTaskSpec):
    def __init__(self,
                 task: Union[type, Callable],
                 task_kwargs: Optional[Dict[str, Any]] = None,
                 name: Optional[str] = None,
                 reduce: Optional[bool] = None,
                 publishes: Optional[List[str]] = None,
                 expects: Optional[List[str]] = None):
        """
        A class to hold specification of a plain task

        Args:
            task (Union[type, Callable]): task class
            task_kwargs (Optional[Dict[str, Any]], optional): task arguments that are used while instantiating.
                                                              Defaults to None.
            name (Optional[str], optional): an unique name of the class. Defaults to None.
            reduce (Optional[bool], optional): a boolean indicating whether this is a reduce task. Defaults to None.
            publishes (Optional[List[str]], optional): a list of result names that this task publishes. 
                                                    Defaults to None.
            expects (Optional[List[str]], optional):  a list of result names that this task expects. Defaults to None.
        """
        super().__init__(task=task, name=name, reduce=reduce,
                         publishes=publishes, expects=expects)
        self.task_kwargs = task_kwargs if task_kwargs is not None else {}

    def build(self) -> List[Task]:
        task = self._create_task_object(task_kwargs=self.task_kwargs)
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
                 gs_expansion_method: Optional[str] = 'product',
                 name: Optional[str] = None,
                 publishes: Optional[List[str]] = None,
                 expects: Optional[List[str]] = None):
        """
        A class to hold specification of a grid searcheable task

        Args:
            task (Union[type, Callable]): task class
            gs_config (Dict[str, Any]): a grid search config that will be expanded
            name (Optional[str], optional): an unique name of the class. Defaults to None.
           publishes (Optional[List[str]], optional): a list of result names that this task publishes. Defaults to None.
            expects (Optional[List[str]], optional):  a list of result names that this task expects. Defaults to None.
        """
        super().__init__(task=task, name=name, publishes=publishes, expects=expects)
        self.task_configs: List[Dict] = GridTaskSpec._split_gs_config(config_grid_search=gs_config,
                                                                      method=gs_expansion_method)

    def build(self) -> List[Task]:
        tasks = [self._create_task_object(
            task_kwargs=config) for config in self.task_configs]
        return tasks

    @staticmethod
    def _find_list_in_dict(obj: Dict, param_grid: List) -> List:
        for key in obj:
            if isinstance(obj[key], list):
                param_grid.append([val for val in obj[key]])
            elif isinstance(obj[key], dict):
                GridTaskSpec._find_list_in_dict(obj[key], param_grid)
            else:
                continue
        return param_grid

    @staticmethod
    def _replace_list_in_dict(obj: Dict, obj_copy: Dict, comb: Tuple, counter: List) -> Tuple[Dict, List]:
        for key, key_copy in zip(obj, obj_copy):
            if isinstance(obj[key], list):
                obj_copy[key_copy] = comb[len(counter)]
                counter.append(1)
            elif isinstance(obj[key], dict):
                GridTaskSpec._replace_list_in_dict(
                    obj[key], obj_copy[key_copy], comb, counter)
            else:
                continue
        return obj_copy, counter

    @staticmethod
    def _split_gs_config(config_grid_search: Dict, method: str = 'product') -> List[Dict]:
        param_grid = []
        param_grid = GridTaskSpec._find_list_in_dict(config_grid_search, param_grid)

        if method == 'product':
            expansion_fn = product
        elif method == 'zip':
            # get the maximum parameter list lengths in config
            max_param_list_len = max([len(param_list) for param_list in param_grid])
            # if a parameter list holds only one element, repeat it max_param_list_len times.
            param_grid = [x * max_param_list_len if len(x) == 1 else x for x in param_grid]
            # check that all parameter grid lists are of same lengths
            if not all(len(param_grid[0]) == len(x) for x in param_grid[1:]):
                raise GridSearchExpansionError('For method "zip" all expanded lists have to be of equal lengths.')
            expansion_fn = zip
        else:
            raise GridSearchExpansionError(f'Expansion method "{method}" is not supported. '
                                           f'Grid search config can only be expanded via "product" or "zip".')

        config_copy = deepcopy(config_grid_search)
        individual_configs = []
        for comb in expansion_fn(*param_grid):
            counter = []
            individual_config = GridTaskSpec._replace_list_in_dict(
                config_grid_search, config_copy, comb, counter)[0]
            individual_config = deepcopy(individual_config)
            individual_configs.append(individual_config)
        return individual_configs
