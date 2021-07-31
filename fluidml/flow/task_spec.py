from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
import inspect
from itertools import product
import json
from typing import Dict, Any, Optional, List, Tuple, Union, Callable, Sequence, Iterable

from fluidml.common import Task, DependencyMixin
from fluidml.common.utils import MyTask
from fluidml.common.exception import GridSearchExpansionError


class BaseTaskSpec(DependencyMixin, ABC):
    def __init__(self,
                 task: Union[type, Callable],
                 name: Optional[str] = None,
                 reduce: Optional[bool] = None,
                 publishes: Optional[List[str]] = None,
                 expects: Optional[List[str]] = None,
                 additional_kwargs: Optional[Dict[str, Any]] = None):
        DependencyMixin.__init__(self)
        self.task = task
        self.name = name if name is not None else self.task.__name__
        self.reduce = reduce
        self.publishes = publishes
        self.expects = expects
        self.additional_kwargs = additional_kwargs if additional_kwargs is not None else {}

        # this will be overwritten later for expanded tasks (a unique expansion id is added)
        self.unique_name = self.name

    def _create_task_object(self,
                            config_kwargs: Dict[str, Any]) -> Task:
        if inspect.isclass(self.task):
            task = self.task(**config_kwargs, **self.additional_kwargs)
            task.config_kwargs = config_kwargs
            expected_inputs = list(inspect.signature(task.run).parameters)

        elif inspect.isfunction(self.task):
            task = MyTask(task=self.task, config_kwargs=config_kwargs, additional_kwargs=self.additional_kwargs)

            task_all_arguments = list(inspect.signature(self.task).parameters)
            task_extra_arguments = list(config_kwargs) + list(self.additional_kwargs) + ['task']
            expected_inputs = [arg for arg in task_all_arguments if arg not in task_extra_arguments]
        else:
            raise TypeError(f'{self.task} needs to be a Class object (type="type") or a function.'
                            f'But it is of type "{type(self.task)}".')
        task.name = self.name
        task.unique_name = self.unique_name

        # override publishes from task spec
        task = self._override_publishes(task)

        # set task expects attribute based on task type and user provided expected inputs
        task = self._set_task_expects(task, expected_inputs)

        return task

    def _set_task_expects(self, task: Task, expected_inputs: List[str]) -> Task:

        if self.expects is not None:
            task.expects = self.expects
        elif task.expects is None and not self.reduce:
            task.expects = expected_inputs

        # if task.expects is None, we collect all published results from each predecessor
        #  and pack them in the "reduced_results" dict.
        return task

    def _override_publishes(self, task: Task) -> Task:
        if self.publishes is not None:
            task.publishes = self.publishes

        elif task.publishes is None:
            task.publishes = []
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
                 config: Optional[Dict[str, Any]] = None,
                 additional_kwargs: Optional[Dict[str, Any]] = None,
                 name: Optional[str] = None,
                 reduce: Optional[bool] = None,
                 publishes: Optional[List[str]] = None,
                 expects: Optional[List[str]] = None):
        """
        A class to hold specification of a plain task

        Args:
            task (Union[type, Callable]): task class
            config (Optional[Dict[str, Any]], optional): task configuration parameters that are used
                                                         while instantiating. Defaults to None.
            additional_kwargs (Optional[Dict[str, Any]], optional): Additional kwargs provided to the task.
            name (Optional[str], optional): an unique name of the class. Defaults to None.
            reduce (Optional[bool], optional): a boolean indicating whether this is a reduce task. Defaults to None.
            publishes (Optional[List[str]], optional): a list of result names that this task publishes. 
                                                    Defaults to None.
            expects (Optional[List[str]], optional):  a list of result names that this task expects. Defaults to None.
        """
        super().__init__(task=task, name=name, reduce=reduce,
                         publishes=publishes, expects=expects, additional_kwargs=additional_kwargs)
        # we assure that the provided config is json serializable since we use json to later store the config
        self.config_kwargs = json.loads(json.dumps(config)) if config is not None else {}

    def build(self) -> List[Task]:
        task = self._create_task_object(config_kwargs=self.config_kwargs)
        return [task]


class GridTaskSpec(BaseTaskSpec):

    def __init__(self,
                 task: Union[type, Callable],
                 gs_config: Dict[str, Any],
                 additional_kwargs: Optional[Dict[str, Any]] = None,
                 gs_expansion_method: Optional[str] = 'product',
                 name: Optional[str] = None,
                 publishes: Optional[List[str]] = None,
                 expects: Optional[List[str]] = None):
        """
        A class to hold specification of a grid searcheable task

        Args:
            task (Union[type, Callable]): task class
            gs_config (Dict[str, Any]): a grid search config that will be expanded
            additional_kwargs (Optional[Dict[str, Any]], optional): Additional kwargs provided to the task.
            name (Optional[str], optional): an unique name of the class. Defaults to None.
            publishes (Optional[List[str]], optional): a list of result names that this task publishes.
                                                       Defaults to None.
            expects (Optional[List[str]], optional):  a list of result names that this task expects. Defaults to None.
        """
        super().__init__(task=task, name=name, publishes=publishes, expects=expects,
                         additional_kwargs=additional_kwargs)
        # we assure that the provided config is json serializable since we use json to later store the config
        gs_config = json.loads(json.dumps(gs_config))
        self.task_configs: List[Dict] = GridTaskSpec._split_gs_config(config_grid_search=gs_config,
                                                                      method=gs_expansion_method)

    def build(self) -> List[Task]:
        tasks = [self._create_task_object(config_kwargs=config) for config in self.task_configs]
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
    def _convert_list_to_tuple_recursively(obj: Sequence) -> Sequence:
        if isinstance(obj, list):
            return tuple(GridTaskSpec._convert_list_to_tuple_recursively(elem) for elem in obj)
        elif isinstance(obj, (set, tuple)):
            return type(obj)(GridTaskSpec._convert_list_to_tuple_recursively(elem) for elem in obj)
        elif isinstance(obj, dict):
            return {k: GridTaskSpec._convert_list_to_tuple_recursively(v) for k, v in obj.items()}
        else:
            return obj

    @staticmethod
    def _get_dict_obj(keys: Iterable, values: Iterable) -> Dict:
        return {key: value for key, value in zip(keys, values)}

    @staticmethod
    def _expand_zip(param_grid: List[List[Any]], keys: List[str]):
        max_param_list_len = max([len(param_list) for param_list in param_grid])
        # if a parameter list holds only one element, repeat it max_param_list_len times.
        param_grid = [x * max_param_list_len if len(x) == 1 else x for x in param_grid]
        # check that all parameter grid lists are of same lengths
        if not all(len(param_grid[0]) == len(x) for x in param_grid[1:]):
            raise ValueError('For method "zip" all expanded lists have to be of equal lengths.')

        return [GridTaskSpec._get_dict_obj(keys, value) for value in zip(*param_grid)]

    @staticmethod
    def _expand_product(param_grid: List[List[Any]], keys: List[str]):
        return [GridTaskSpec._get_dict_obj(keys, value) for value in product(*param_grid)]

    @staticmethod
    def _expand(splits_by_keys: Dict, method: str = 'product') -> List[Dict]:
        param_grid = list(splits_by_keys.values())
        keys = list(splits_by_keys.keys())

        if method == 'zip':
            exp_configs: List[Dict] = GridTaskSpec._expand_zip(param_grid, keys)
        elif method == 'product':
            exp_configs: List[Dict] = GridTaskSpec._expand_product(param_grid, keys)
        else:
            raise GridSearchExpansionError(f'Expansion method "{method}" is not supported. '
                                           f'Grid search config can only be expanded via "product" or "zip".')
        return exp_configs

    @staticmethod
    def _split_config_groups(obj: Dict, method: str = 'product') -> Union[Any, List[Dict]]:
        """
        Recursively splits a dict on lists that contain (nested) dicts as list elements.
        """
        if isinstance(obj, dict):
            splits_per_key = {key: GridTaskSpec._split_config_groups(child_obj, method)
                              for key, child_obj in obj.items()}
            return GridTaskSpec._expand(splits_per_key, method)
        elif isinstance(obj, list):
            if any(isinstance(elem, dict) for elem in obj):
                return [splits for item in obj for splits in GridTaskSpec._split_config_groups(item, method)]
            else:
                return [obj]
        else:
            return [obj]

    @staticmethod
    def _split_gs_config(config_grid_search: Dict, method: str = 'product') -> List[Dict]:

        config_groups = GridTaskSpec._split_config_groups(config_grid_search, method)

        individual_configs = []
        for config_grid_search in config_groups:
            param_grid = []
            param_grid = GridTaskSpec._find_list_in_dict(config_grid_search, param_grid)

            # wrap empty parameter lists in a seperate list
            # -> the outer list will be expanded and
            #    the inner empty list will be passed as a single argument to all task instances.
            param_grid = [x if x else [x] for x in param_grid]

            if method == 'product':
                # get unique zip identifier and their respective parameter combination ids in the param_grid list
                zip_identifier = defaultdict(lambda: defaultdict(list))
                for i, var_params in enumerate(param_grid):
                    if isinstance(var_params[-1], str) and var_params[-1].startswith('@'):
                        zip_identifier[var_params[-1]]['param_comb_ids'].append(i)

                # remove special zip_identifier ('@<identifier>') from param grid
                param_grid = [var_params[:-1]
                              if isinstance(var_params[-1], str) and var_params[-1].startswith('@')
                              else var_params
                              for var_params in param_grid]

                # store zipped param combinations for each zip identifier
                for identifier, values in zip_identifier.items():
                    zip_identifier[identifier]['zipped'] = list(
                        zip(*[param_grid[idx] for idx in values['param_comb_ids']]))

                # create all param combinations
                all_combs = list(product(*param_grid))

                # remove all param combinations that violate one of the zipped combinations
                combinations = []
                unique_combinations = set()
                for comb in all_combs:
                    add_comb = True
                    for identifier in zip_identifier.values():
                        if tuple(comb[i] for i in identifier['param_comb_ids']) not in identifier['zipped']:
                            add_comb = False
                            break
                    if add_comb:
                        tuple_comb = GridTaskSpec._convert_list_to_tuple_recursively(comb)
                        if tuple_comb not in unique_combinations:
                            combinations.append(comb)
                            unique_combinations.add(tuple_comb)

            elif method == 'zip':
                param_grid = [var_params[:-1]
                              if isinstance(var_params[-1], str) and var_params[-1].startswith('@')
                              else var_params
                              for var_params in param_grid]

                # get the maximum parameter list lengths in config
                max_param_list_len = max([len(param_list) for param_list in param_grid])
                # if a parameter list holds only one element, repeat it max_param_list_len times.
                param_grid = [x * max_param_list_len if len(x) == 1 else x for x in param_grid]
                # check that all parameter grid lists are of same lengths
                if not all(len(param_grid[0]) == len(x) for x in param_grid[1:]):
                    raise GridSearchExpansionError('For method "zip" all expanded lists have to be of equal lengths.')

                combinations = zip(*param_grid)

            else:
                raise GridSearchExpansionError(f'Expansion method "{method}" is not supported. '
                                               f'Grid search config can only be expanded via "product" or "zip".')

            config_copy = deepcopy(config_grid_search)

            for comb in combinations:
                counter = []
                individual_config = GridTaskSpec._replace_list_in_dict(
                    config_grid_search, config_copy, comb, counter)[0]
                individual_config = deepcopy(individual_config)
                individual_configs.append(individual_config)
        return individual_configs
