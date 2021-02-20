from collections import defaultdict
import logging
from itertools import product
from typing import List, Any, Dict, Optional

from networkx import DiGraph
from networkx.algorithms.dag import topological_sort

from fluidml.common import Task
from fluidml.common.utils import update_merge, reformat_config
from fluidml.common.exception import NoTasksError
from fluidml.flow import BaseTaskSpec, GridTaskSpec
from fluidml.swarm import Swarm


logger = logging.getLogger(__name__)


class Flow:
    """A class that implements the core logic of building tasks from task specifications

    - It automatically expands the tasks based on task spec and task config
    - It extends the dependencies to the expanded tasks
    - Finally, it composes a list of tasks which are then run through the provided swarm
    """

    def __init__(self,
                 swarm: Swarm,
                 task_to_execute: Optional[str] = None,
                 force: Optional[str] = None):
        """
        Args:
            swarm (Swarm): an instance of the swarm
            task_to_execute (Optional[str], optional): a list of task names only those tasks which needs to be run 
            force (Optional[str], optional): forcefully re-run tasks
                Possible options are:
                    "selected" - Only specified tasks in task_to_execture are re-run
                    "all" - All the tasks are re-run
        """
        self._swarm = swarm
        self._task_to_execute = task_to_execute
        self._force = force

    @staticmethod
    def _create_task_spec_graph(task_specs: List[BaseTaskSpec]) -> DiGraph:
        task_spec_graph = DiGraph()
        for spec in task_specs:
            task_spec_graph.add_node(spec.name)
            for predecessor in spec.predecessors:
                task_spec_graph.add_edge(predecessor.name, spec.name)
        return task_spec_graph

    def _register_tasks_to_force_execute(self, tasks: List[Task]):
        if self._force == 'selected':
            assert self._task_to_execute is not None, '"task_to_execute" is required for force = "selected".'
            for task in tasks:
                if task.name == self._task_to_execute:
                    task.force = True
        elif self._force == 'all':
            for task in tasks:
                task.force = True
        elif self._force is not None:
            raise ValueError(
                f'force = {self._force} is not supported. Choose "all" or "selected"')

    def _order_task_specs(self,
                          task_specs: List[BaseTaskSpec]) -> List[BaseTaskSpec]:
        # task spec graph holding the user defined dependency structure
        task_spec_graph: DiGraph = Flow._create_task_spec_graph(
            task_specs=task_specs)

        # topological ordering of tasks in graph
        # if a specific task to execute is provided, remove non dependent tasks from graph
        if self._task_to_execute:
            sorted_names = list(topological_sort(task_spec_graph))
            target_idx = sorted_names.index(self._task_to_execute)
            sorted_names = sorted_names[:target_idx + 1]
        else:
            sorted_names = list(topological_sort(task_spec_graph))

        # convert task_specs to dict, so it can be queried by name
        name_to_task_spec = {spec.name: spec for spec in task_specs}

        # get sorted list of task specs
        sorted_specs = [name_to_task_spec[name] for name in sorted_names]
        return sorted_specs

    @staticmethod
    def _validate_task_combination(task_combination: List[Task]) -> bool:
        def _match(task_cfgs: List[Dict[str, Any]]):
            unique_cfgs = []
            for config in task_cfgs:
                if config not in unique_cfgs:
                    unique_cfgs.append(config)

            if len(unique_cfgs) == 1:
                return True
            return False

        # we validate the task combinations based on their path in the task graph
        # if two tasks have same parent task and their configs are different
        # then the combination is not valid
        task_names_in_path = list(
            set(name for task in task_combination for name in task.unique_config))

        # for each task in path config
        for name in task_names_in_path:
            # task configs
            task_configs = [task.unique_config[name] for task in task_combination
                            if name in task.unique_config.keys()]

            # if they do not match, return False
            if not _match(task_configs):
                return False
        return True

    @staticmethod
    def _get_predecessor_product(expanded_tasks_by_name: Dict[str, List[Task]],
                                 task_spec: BaseTaskSpec) -> List[List[Task]]:
        predecessor_tasks = [expanded_tasks_by_name[predecessor.name]
                             for predecessor in task_spec.predecessors]
        task_combinations = [list(item) for item in product(
            *predecessor_tasks)] if predecessor_tasks else [[]]
        task_combinations = [combination for combination in task_combinations
                             if Flow._validate_task_combination(combination)]
        return task_combinations

    @staticmethod
    def _combine_task_config(tasks: List[Task]):
        config = {}
        for task in tasks:
            config = {**config, **task.unique_config}
        return config

    @staticmethod
    def _merge_task_combination_configs(task_combinations: List[List[Task]], task_specs: List[BaseTaskSpec]) -> Dict:
        task_configs = [
            task.unique_config for combination in task_combinations for task in combination]
        merged_config = task_configs.pop(0)
        for config in task_configs:
            merged_config: Dict = update_merge(merged_config, config)

        if task_configs:
            # get all task names that were specified as GridTaskSpec
            grid_task_names = [
                spec.name for spec in task_specs if isinstance(spec, GridTaskSpec)]

            # split merged_config in grid_task_config and normal_task_config
            grid_task_config = {
                key: value for key, value in merged_config.items() if key in grid_task_names}
            normal_task_config = {
                key: value for key, value in merged_config.items() if key not in grid_task_names}

            # reformat only grid_task_config (replace tuples by lists)
            grid_task_config: Dict = reformat_config(grid_task_config)

            # merge back the normal_task_config with the formatted grid_task_config
            merged_config = {**normal_task_config, **grid_task_config}
        return merged_config

    @staticmethod
    def _generate_tasks(task_specs: List[BaseTaskSpec]) -> List[Task]:
        # keep track of expanded tasks by their names
        expanded_tasks_by_name = defaultdict(list)
        task_id = 0

        # for each task to expand
        for exp_task in task_specs:
            # get predecessor task combinations
            task_combinations = Flow._get_predecessor_product(
                expanded_tasks_by_name, exp_task)

            if exp_task.reduce:
                # if it is a reduce task, just add the predecessor task
                # combinations as parents
                tasks = exp_task.build()

                for task in tasks:
                    # predecessor config
                    predecessor_config = Flow._merge_task_combination_configs(
                        task_combinations, task_specs)

                    # add dependencies
                    for task_combination in task_combinations:
                        task.requires(task_combination)

                    task.id_ = task_id
                    task.reduce = True
                    expanded_tasks_by_name[task.name].append(task)
                    task.unique_config = {
                        **predecessor_config, **{task.name: task.kwargs}}
                    task_id += 1
            else:
                # for each combination, create a new task
                for task_combination in task_combinations:
                    tasks = exp_task.build()

                    # shared predecessor config
                    predecessor_config = Flow._combine_task_config(
                        task_combination)

                    # for each task that is created, add ids and dependencies
                    for task in tasks:
                        task.id_ = task_id
                        task.requires(task_combination)
                        task.unique_config = {
                            **predecessor_config, **{task.name: task.kwargs}}
                        expanded_tasks_by_name[task.name].append(task)
                        task_id += 1

        # create final list of tasks
        tasks = [task for expanded_tasks in expanded_tasks_by_name.values()
                 for task in expanded_tasks]
        return tasks

    def run(self,
            task_specs: List[BaseTaskSpec]) -> Dict[str, Dict[str, Any]]:
        """Runs the specified tasks and returns the results

        Args:
            task_specs (List[Union[TaskSpec, GridTaskSpec]]): list of task specifications

        Returns:
            Dict[str, Dict[str, Any]]: a nested dict of results

        """
        if not task_specs:
            raise NoTasksError("There are no tasks to run")

        ordered_task_specs = self._order_task_specs(task_specs)
        tasks = Flow._generate_tasks(ordered_task_specs)
        self._register_tasks_to_force_execute(tasks)
        results = self._swarm.work(tasks)
        return results
