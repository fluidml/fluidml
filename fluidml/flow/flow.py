from collections import defaultdict
from itertools import product
from typing import List, Any, Dict, Optional, Tuple, Union

from networkx import DiGraph, shortest_path_length
from networkx.algorithms.dag import topological_sort

from fluidml.common import Task
from fluidml.flow.task_spec import BaseTaskSpec, GridTaskSpec, TaskSpec
from fluidml.swarm import Swarm


class Flow:
    """A class that implements the core logic of building tasks from task specifications

    - It automatically expands the tasks based on task spec and task config
    - It extends the dependencies to the expanded tasks
    - Finally, it composes a list of tasks which are then run through the provided swarm
    """

    def __init__(self,
                 swarm: Swarm,
                 task_to_execute: Optional[str] = None):
        self._swarm = swarm
        self._task_to_execute = task_to_execute

    @staticmethod
    def _create_task_spec_graph(task_specs: List[BaseTaskSpec]) -> DiGraph:
        task_spec_graph = DiGraph()
        for spec in task_specs:
            for predecessor in spec.predecessors:
                task_spec_graph.add_edge(predecessor.name, spec.name)
        return task_spec_graph

    @staticmethod
    def _create_single_run_configs(task_specs: List[BaseTaskSpec]) -> List[Dict]:
        name_to_params = defaultdict(list)
        for spec in task_specs:
            if isinstance(spec, GridTaskSpec):
                task_configs = spec.task_configs
            elif isinstance(spec, TaskSpec):
                task_configs = spec.task_kwargs
            else:
                raise ValueError('Object spec has to be of instance type GridTaskSpec or TaskSpec.')
            name_to_params[spec.name].extend(task_configs)

        # TODO: do we need this unit test?
        # in case no tasks are provided to flow
        if name_to_params:
            task_names, values = zip(*name_to_params.items())
            single_run_configs = [dict(zip(task_names, params)) for params in product(*values)]
        else:
            single_run_configs = [{}]
        return single_run_configs

    def _order_task_specs(self,
                          task_specs: List[BaseTaskSpec]) -> List[BaseTaskSpec]:
        # task spec graph holding the user defined dependency structure
        task_spec_graph: DiGraph = Flow._create_task_spec_graph(task_specs=task_specs)

        # topological ordering of tasks in graph
        # if a specific task to execute is provided, remove non dependent tasks from graph
        if self._task_to_execute:
            sorted_names = list(shortest_path_length(task_spec_graph, target=self._task_to_execute).keys())[::-1]
        else:
            sorted_names = list(topological_sort(task_spec_graph))

        # convert task_specs to dict, so it can be queried by name
        name_to_task_spec = {spec.name: spec for spec in task_specs}

        # get sorted list of task specs
        sorted_specs = [name_to_task_spec[name] for name in sorted_names]
        return sorted_specs

    @staticmethod
    def _update_merge(d1: Dict, d2: Dict) -> Union[Dict, Tuple]:
        if isinstance(d1, dict) and isinstance(d2, dict):
            # Unwrap d1 and d2 in new dictionary to keep non-shared keys with **d1, **d2
            # Next unwrap a dict that treats shared keys
            # If two keys have an equal value, we take that value as new value
            # If the values are not equal, we recursively merge them
            return {**d1, **d2,
                    **{k: d1[k] if d1[k] == d2[k] else Flow._update_merge(d1[k], d2[k])
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

    @staticmethod
    def _reformat_config(d: Dict) -> Dict:
        for key, value in d.items():
            if isinstance(value, list):
                d[key] = [value]
            if isinstance(value, tuple):
                d[key] = list(value)
            elif isinstance(value, dict):
                Flow._reformat_config(value)
            else:
                continue
        return d

    @staticmethod
    def _merge_configs(configs: List[Dict]) -> Dict:
        merged_config = configs.pop(0)
        for config in configs:
            merged_config: Dict = Flow._update_merge(merged_config, config)

        merged_config: Dict = Flow._reformat_config(merged_config)
        return merged_config

    # @staticmethod
    # def _get_predecessor_product(expanded_tasks_by_name: Dict[str, List[Task]],
    #                              task_spec: BaseTaskSpec) -> List[List[Task]]:
    #     predecessor_tasks = [expanded_tasks_by_name[predecessor.name] for predecessor in task_spec.predecessors]
    #     task_combinations = [list(item) for item in product(*predecessor_tasks)] if predecessor_tasks else [[]]
    #     return task_combinations

    # @staticmethod
    # def _generate_tasks(task_specs: List[BaseTaskSpec]) -> List[Task]:
    #     # keep track of expanded tasks by their names
    #     expanded_tasks_by_name = defaultdict(list)
    #     task_id = 0
    #
    #     # for each task to expand
    #     for exp_task in task_specs:
    #         # get predecessor task combinations
    #         task_combinations = Flow._get_predecessor_product(expanded_tasks_by_name, exp_task)
    #
    #         if exp_task.reduce:
    #             tasks = exp_task.build()
    #             for task in tasks:
    #                 for task_combination in task_combinations:
    #                     task.requires(task_combination)
    #                 task.id_ = task_id
    #                 expanded_tasks_by_name[task.name].append(task)
    #                 task_id += 1
    #         else:
    #             # for each combination, create a new task
    #             for task_combination in task_combinations:
    #                 tasks = exp_task.build()
    #
    #                 # for each task that is created, add ids and dependencies
    #                 for task in tasks:
    #                     task.id_ = task_id
    #                     task.requires(task_combination)
    #                     expanded_tasks_by_name[task.name].append(task)
    #                     task_id += 1
    #
    #     # create final list of tasks
    #     tasks = [task for expanded_tasks in expanded_tasks_by_name.values() for task in expanded_tasks]
    #     return tasks

    @staticmethod
    def _generate_tasks(task_specs: List[BaseTaskSpec]) -> List[Task]:
        single_run_configs = Flow._create_single_run_configs(task_specs=task_specs)

        task_spec_graph: DiGraph = Flow._create_task_spec_graph(task_specs=task_specs)

        expanded_tasks = []
        task_id = 0
        for exp_task in task_specs:
            if not exp_task.reduce:
                for config in single_run_configs:
                    tasks = exp_task.build()
                    for task in tasks:
                        ancestor_task_spec_names = list(shortest_path_length(task_spec_graph,
                                                                             target=task.name).keys())[::-1]

                        task.unique_config = {name: config[name] for name in ancestor_task_spec_names}

                        # check if task object is already included in task dictionary, skip if yes
                        if task.unique_config in [task_.unique_config for task_ in expanded_tasks]:
                            continue

                        pred_names = [pred.name for pred in exp_task.predecessors]

                        pred_tasks = [pred_task for pred_task in expanded_tasks
                                      if pred_task.unique_config.items() < task.unique_config.items()
                                      and pred_task.name in pred_names]

                        task.requires(pred_tasks)
                        task.id_ = task_id
                        task_id += 1
                        expanded_tasks.append(task)

            else:
                tasks = exp_task.build()
                for task in tasks:
                    pred_names = [pred.name for pred in exp_task.predecessors]
                    pred_tasks = [pred_task for pred_task in expanded_tasks if pred_task.name in pred_names]
                    task.requires(pred_tasks)

                    task_configs = []
                    for pred in task.predecessors:
                        pred_unique_conf = pred.unique_config
                        pred_unique_conf[task.name] = task.kwargs
                        task_configs.append(pred_unique_conf)

                    task.unique_config = Flow._merge_configs(task_configs)

                    task.id_ = task_id
                    task_id += 1
                    expanded_tasks.append(task)

        return expanded_tasks

    def run(self,
            task_specs: List[BaseTaskSpec]) -> Dict[str, Dict[str, Any]]:
        """Runs the specified tasks and returns the results

        Args:
            task_specs (List[Union[TaskSpec, GridTaskSpec]]): list of task specifications

        Returns:
            Dict[str, Dict[str, Any]]: a nested dict of results

        """

        ordered_task_specs = self._order_task_specs(task_specs)
        tasks = Flow._generate_tasks(ordered_task_specs)
        results = self._swarm.work(tasks)
        return results
