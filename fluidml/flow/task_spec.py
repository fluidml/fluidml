import inspect
import json
from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from itertools import product
from typing import Dict, Any, Optional, List, Tuple, Union, Callable, Sequence, Iterable, Type

from metadict import MetaDict

from fluidml.common import DependencyMixin, Task, Resource
from fluidml.common.exception import GridSearchExpansionError
from fluidml.storage import ResultsStore


class BaseTaskSpec(DependencyMixin, ABC):
    def __init__(
        self,
        task: Union[Type[Task], Callable],
        name: Optional[str] = None,
        publishes: Optional[List[str]] = None,
        expects: Optional[List[str]] = None,
        reduce: Optional[bool] = None,
        additional_kwargs: Optional[Union[Dict[str, Any], MetaDict]] = None,
    ):
        DependencyMixin.__init__(self)

        # task has to be a class object which inherits Task or it has to be a function
        if not (self._inherits_from_task_class(task) or inspect.isfunction(task)):
            raise TypeError(
                f'{task} needs to be a Class object which inherits Task (type="type") or a function.'
                f'But it is of type "{type(task)}".'
            )
        self.task = task
        self.name = name if name is not None else self.task.__name__
        if publishes is None and self._inherits_from_task_class(task):
            publishes = task.publishes if hasattr(task, "publishes") else []
        self.publishes = publishes
        self.expects = expects
        self.reduce = reduce

        self.additional_kwargs = additional_kwargs if additional_kwargs is not None else {}

        # this will be overwritten later for expanded tasks (a unique expansion id is added)
        self._unique_name = self.name

    @staticmethod
    def _inherits_from_task_class(task: Union[Type[Task], Callable]) -> bool:
        return (
            inspect.isclass(task)
            and f"{task.__base__.__module__}.{task.__base__.__name__}" == f"{Task.__module__}.{Task.__name__}"
        )

    @property
    def unique_name(self):
        return self._unique_name

    @unique_name.setter
    def unique_name(self, unique_name: str):
        self._unique_name = unique_name

    @abstractmethod
    def expand(self) -> List["TaskSpec"]:
        raise NotImplementedError


class TaskSpec(BaseTaskSpec):
    def __init__(
        self,
        task: Union[Type[Task], Callable],
        config: Optional[Union[Dict[str, Any], MetaDict]] = None,
        additional_kwargs: Optional[Union[Dict[str, Any], MetaDict]] = None,
        name: Optional[str] = None,
        reduce: Optional[bool] = None,
        publishes: Optional[List[str]] = None,
        expects: Optional[List[str]] = None,
        resource: Optional[Resource] = None,
    ):
        """A class to hold specification of a plain task.

        Args:
            task: Task class
            config: Task configuration parameters that are used while instantiating. Defaults to ``None``.
            additional_kwargs: Additional kwargs provided to the task.
            name: A unique name of the class. Defaults to None.
            reduce: A boolean indicating whether this is a reduce task. Defaults to None.
            publishes: A list of result names that this task publishes. Defaults to None.
            expects: A list of result names that this task expects. Defaults to None.
            resource: An optional ``Resource`` object assigned to a task.
                If resources are assigned to workers via ``Swarm`` this argument has no effect.
        """
        # todo: Remove additional kwargs from config via prefix and add to additional kwargs dict
        #  ...
        super().__init__(task, name, publishes, expects, reduce, additional_kwargs=additional_kwargs)
        # we assure that the provided config is json serializable since we use json to later store the config
        self.config_kwargs = json.loads(json.dumps(config)) if config is not None else {}

        # set in Flow or Swarm (if pipeline is executed via swarm) or manually
        self._project_name: Optional[str] = None
        self._run_name: Optional[str] = None
        self._results_store: Optional[ResultsStore] = None
        self._resource: Optional[Resource] = resource

        # set in Flow
        self._id: Optional[int] = None
        self._unique_config: Optional[Dict] = None
        self._force: Optional[str] = None

    @property
    def id_(self):
        return self._id

    @id_.setter
    def id_(self, id_: int):
        self._id = id_

    @property
    def unique_config(self):
        return self._unique_config

    @unique_config.setter
    def unique_config(self, config: Dict):
        self._unique_config = config

    @property
    def force(self):
        return self._force

    @force.setter
    def force(self, force: str):
        self._force = force

    @property
    def project_name(self):
        return self._project_name

    @project_name.setter
    def project_name(self, project_name: str):
        self._project_name = project_name

    @property
    def run_name(self):
        return self._run_name

    @run_name.setter
    def run_name(self, run_name: str):
        self._run_name = run_name

    @property
    def results_store(self):
        return self._results_store

    @results_store.setter
    def results_store(self, results_store: ResultsStore):
        self._results_store = results_store

    @property
    def resource(self):
        return self._resource

    @resource.setter
    def resource(self, resource: Resource):
        self._resource = resource

    def expand(self) -> List["TaskSpec"]:
        # we don't return [self] since we have to return a copy without self.predecessors and self.successors
        return [
            TaskSpec(
                task=self.task,
                config=self.config_kwargs,
                additional_kwargs=self.additional_kwargs,
                name=self.name,
                reduce=self.reduce,
                publishes=self.publishes,
                expects=self.expects,
                resource=self.resource,
            )
        ]


class GridTaskSpec(BaseTaskSpec):
    def __init__(
        self,
        task: Union[Type[Task], Callable],
        gs_config: Optional[Union[Dict[str, Any], MetaDict]],
        additional_kwargs: Optional[Union[Dict[str, Any], MetaDict]] = None,
        gs_expansion_method: Optional[str] = "product",
        name: Optional[str] = None,
        publishes: Optional[List[str]] = None,
        expects: Optional[List[str]] = None,
        resource: Optional[Resource] = None,
    ):
        """A class to hold specification of a grid-search expandable task

        Args:
            task: Task class
            gs_config: A grid search config that will be expanded
            additional_kwargs: Additional kwargs provided to the task.
            name: A unique name of the class. Defaults to None.
            publishes: A list of result names that this task publishes. Defaults to None.
            expects: A list of result names that this task expects. Defaults to None.
            resource: An optional ``Resource`` object assigned to a task.
                If resources are assigned to workers via ``Swarm`` this argument has no effect.
        """
        super().__init__(task, name, publishes, expects, additional_kwargs=additional_kwargs)

        # we assure that the provided config is json serializable since we use json to later store the config
        gs_config = json.loads(json.dumps(gs_config))
        self.task_configs: List[Dict] = GridTaskSpec._split_gs_config(
            config_grid_search=gs_config, method=gs_expansion_method
        )

        self.resource = resource

    def expand(self) -> List["TaskSpec"]:
        return [
            TaskSpec(
                task=self.task,
                config=config,
                additional_kwargs=self.additional_kwargs,
                name=self.name,
                publishes=self.publishes,
                expects=self.expects,
                resource=self.resource,
            )
            for config in self.task_configs
        ]

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
                GridTaskSpec._replace_list_in_dict(obj[key], obj_copy[key_copy], comb, counter)
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
    def _expand(splits_by_keys: Dict, method: str = "product") -> List[Dict]:
        param_grid = list(splits_by_keys.values())
        keys = list(splits_by_keys.keys())

        if method == "zip":
            exp_configs: List[Dict] = GridTaskSpec._expand_zip(param_grid, keys)
        elif method == "product":
            exp_configs: List[Dict] = GridTaskSpec._expand_product(param_grid, keys)
        else:
            raise GridSearchExpansionError(
                f'Expansion method "{method}" is not supported. '
                f'Grid search config can only be expanded via "product" or "zip".'
            )
        return exp_configs

    @staticmethod
    def _split_config_groups(obj: Dict, method: str = "product") -> Union[Any, List[Dict]]:
        """Recursively splits a dict on lists that contain (nested) dicts as list elements."""
        if isinstance(obj, dict):
            if not obj:  # in case of empty dict -> return dict without expanding
                return [obj]
            splits_per_key = {
                key: GridTaskSpec._split_config_groups(child_obj, method) for key, child_obj in obj.items()
            }
            return GridTaskSpec._expand(splits_per_key, method)
        elif isinstance(obj, list):
            if any(isinstance(elem, dict) for elem in obj):
                return [splits for item in obj for splits in GridTaskSpec._split_config_groups(item, method)]
            else:
                return [obj]
        else:
            return [obj]

    @staticmethod
    def _split_gs_config(config_grid_search: Dict, method: str = "product") -> List[Dict]:

        config_groups = GridTaskSpec._split_config_groups(config_grid_search, method)

        individual_configs = []
        for config_grid_search in config_groups:
            param_grid = []
            param_grid = GridTaskSpec._find_list_in_dict(config_grid_search, param_grid)

            # wrap empty parameter lists in a separate list
            # -> the outer list will be expanded and
            #    the inner empty list will be passed as a single argument to all task instances.
            param_grid = [x if x else [x] for x in param_grid]

            if method == "product":
                # get unique zip identifier and their respective parameter combination ids in the param_grid list
                zip_identifier = defaultdict(lambda: defaultdict(list))
                for i, var_params in enumerate(param_grid):
                    if isinstance(var_params[-1], str) and var_params[-1].startswith("@"):
                        zip_identifier[var_params[-1]]["param_comb_ids"].append(i)

                # remove special zip_identifier ('@<identifier>') from param grid
                param_grid = [
                    var_params[:-1]
                    if isinstance(var_params[-1], str) and var_params[-1].startswith("@")
                    else var_params
                    for var_params in param_grid
                ]

                # store zipped param combinations for each zip identifier
                for identifier, values in zip_identifier.items():
                    zip_identifier[identifier]["zipped"] = list(
                        zip(*[param_grid[idx] for idx in values["param_comb_ids"]])
                    )

                # create all param combinations
                all_combs = list(product(*param_grid))

                # remove all param combinations that violate one of the zipped combinations
                combinations = []
                unique_combinations = set()
                for comb in all_combs:
                    add_comb = True
                    for identifier in zip_identifier.values():
                        if tuple(comb[i] for i in identifier["param_comb_ids"]) not in identifier["zipped"]:
                            add_comb = False
                            break
                    if add_comb:
                        tuple_comb = GridTaskSpec._convert_list_to_tuple_recursively(comb)
                        if tuple_comb not in unique_combinations:
                            combinations.append(comb)
                            unique_combinations.add(tuple_comb)

            elif method == "zip":
                param_grid = [
                    var_params[:-1]
                    if isinstance(var_params[-1], str) and var_params[-1].startswith("@")
                    else var_params
                    for var_params in param_grid
                ]

                # get the maximum parameter list lengths in config
                max_param_list_len = max([len(param_list) for param_list in param_grid])
                # if a parameter list holds only one element, repeat it max_param_list_len times.
                param_grid = [x * max_param_list_len if len(x) == 1 else x for x in param_grid]
                # check that all parameter grid lists are of same lengths
                if not all(len(param_grid[0]) == len(x) for x in param_grid[1:]):
                    raise GridSearchExpansionError('For method "zip" all expanded lists have to be of equal lengths.')

                combinations = zip(*param_grid)

            else:
                raise GridSearchExpansionError(
                    f'Expansion method "{method}" is not supported. '
                    f'Grid search config can only be expanded via "product" or "zip".'
                )

            config_copy = deepcopy(config_grid_search)

            for comb in combinations:
                counter = []
                individual_config = GridTaskSpec._replace_list_in_dict(config_grid_search, config_copy, comb, counter)[
                    0
                ]
                individual_config = deepcopy(individual_config)
                individual_configs.append(individual_config)
        return individual_configs
