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


def get_dict_obj(keys: List, values: List) -> Dict:
    dict = {}
    for key, value in zip(keys, values):
        dict[key] = value
    return dict


def find_products(splits_by_keys: Dict) -> List[Dict]:
    values = list(splits_by_keys.values())
    keys = list(splits_by_keys.keys())
    if len(values) == 1:
        dict_objs = [get_dict_obj(keys, [value]) for value in values[0]]
    else:
        product_values = product(*values)
        dict_objs = [get_dict_obj(keys, value) for value in product_values]
    return dict_objs


def to_expand(obj: Any) -> bool:
    expand = True if isinstance(obj, dict) and obj.get(
        "expand", False) else False
    return expand


def split_config(obj: Dict) -> List[Dict]:
    """
    Recursively splits the given object
    """
    if not isinstance(obj, dict):
        return obj

    # it is a dict and further split
    splits_by_key = {}
    for key, child_obj in obj.items():
        if key != "expand":
            if to_expand(child_obj):
                all_splits = []
                for item in child_obj["values"]:
                    splits = split_config(item)
                    if isinstance(splits, list):
                        all_splits.extend(splits)
                    else:
                        all_splits.append(splits)
                splits_by_key[key] = all_splits

            # another dict, which needs to be expanded
            elif isinstance(child_obj, dict):
                splits_by_key[key] = split_config(child_obj)
            else:  # others which need not be expanded
                splits_by_key[key] = [child_obj]

    # here, find cartesian
    configs = find_products(splits_by_key)

    return configs


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
        self.task_configs: List[Dict] = split_config(gs_config)

    def build(self) -> List[Task]:
        tasks = [self._create_task_object(
            task_kwargs=config) for config in self.task_configs]
        return tasks
