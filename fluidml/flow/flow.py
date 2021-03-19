from collections import defaultdict
import logging
from itertools import product
import sys
from typing import List, Any, Dict, Optional, Set, Union

import networkx as nx
from networkx import DiGraph
from networkx.algorithms.dag import topological_sort
from rich.traceback import install as rich_install

from fluidml.common import Task
from fluidml.common.utils import update_merge, reformat_config
from fluidml.common.exception import NoTasksError, CyclicGraphError, TaskNameError
from fluidml.flow import BaseTaskSpec, GridTaskSpec
from fluidml.flow.graph_visualization import create_console_graph
from fluidml.flow.pager import FluidPager
from fluidml.swarm import Swarm


rich_install(extra_lines=2)
logger = logging.getLogger(__name__)


class Flow:
    """A class that implements the core logic of building tasks from task specifications

    - It automatically expands the tasks based on task spec and task config
    - It extends the dependencies to the expanded tasks
    - It provides the task graph objects and simple console graph visualization
    - Finally, it composes a list of tasks which are then run through the provided swarm
    """

    def __init__(self, swarm: Swarm):
        """
        Args:
            swarm (Swarm): an instance of the swarm
        """
        self._swarm = swarm

        # self._task_to_execute: Optional[str] = None

        # contains the expanded graph as list of Task objects -> used internally in swarm
        self._expanded_tasks: Optional[List[Task]] = None
        # contains the expanded graph as networkx DiGraph -> accessible to the user for introspection and visualization
        self.task_graph: Optional[DiGraph] = None
        # contains the original, user defined task spec graph as networkx DiGraph
        # -> accessible to the user for introspection and visualization
        self.task_spec_graph: Optional[DiGraph] = None

    @staticmethod
    def _check_acyclic(graph: DiGraph) -> None:

        # try to find cycles in graph -> return None if None are found
        try:
            edges = nx.find_cycle(graph, orientation='original')
        except nx.NetworkXNoCycle:
            return

        # gather nodes involved in cycle and raise an error
        nodes_containing_cycle: Set[str] = set()
        for from_node, to_node, _ in edges:
            nodes_containing_cycle.add(from_node)
            nodes_containing_cycle.add(to_node)

        msg = f'Pipeline has a cycle involving: {", ".join(list(nodes_containing_cycle))}.'
        raise CyclicGraphError(msg)

    @staticmethod
    def _check_no_task_name_clash(task_specs: List[BaseTaskSpec]) -> None:
        from importlib import import_module

        for task_spec in task_specs:
            task_obj = task_spec.task
            task_name = task_spec.task.__name__
            module_name = getattr(task_obj, '__module__')
            import_module(module_name)
            module = sys.modules[module_name]
            obj = getattr(module, task_name)
            if obj is not task_obj:
                raise TaskNameError(
                    f'Task names have to be unique. '
                    f'A second object different from task "{task_name}" was found with the same name: \n'
                    f'{obj} in {module}.'
                )

    @staticmethod
    def _create_graph_from_task_list(tasks: List[Union[BaseTaskSpec, Task]], name: Optional[str] = None) -> DiGraph:
        """ Creates nx.DiGraph object of the list of defined tasks with registered dependencies."""
        graph = DiGraph()
        for task in tasks:
            graph.add_node(task.unique_name, task=task)
            for predecessor in task.predecessors:
                graph.add_node(predecessor.unique_name, task=predecessor)
                graph.add_edge(predecessor.unique_name, task.unique_name)
        if name is not None:
            graph.name = name
        return graph

    def _create_task_spec_graph(self, task_specs: List[BaseTaskSpec]) -> DiGraph:
        task_spec_graph = Flow._create_graph_from_task_list(tasks=task_specs, name='task spec graph')

        # assure that task spec graph contains no cyclic dependencies
        Flow._check_acyclic(graph=task_spec_graph)
        self.task_spec_graph = task_spec_graph
        return task_spec_graph

    def _register_tasks_to_force_execute(self, force: Union[str, List[str]]) -> None:
        # if force == 'all' set force to True for all tasks
        if force == 'all':
            for task in self._expanded_tasks:
                task.force = True
            return

        # convert to list if force is of type str
        if isinstance(force, str):
            force = [force]

        if not isinstance(force, List):
            raise TypeError('"force" argument has to be of type str or list of str.')

        # get all user provided task names to force execute
        force_task_names = [task_name[:-1] if task_name[-1] == '+' else task_name for task_name in force]
        # get all task names defined in the task spec graph
        task_names = list(self.task_spec_graph.nodes)
        # find all user provided task names to force execute that don't exist in the task spec graph
        unknown_task_names = list(set(force_task_names).difference(task_names))
        # if any unknown names are found, raise a ValueError
        if unknown_task_names:
            raise ValueError(f'The following task names provided to "force" are unknown: '
                             f'{", ".join(unknown_task_names)}')

        # create a list of unique task names to force execute
        # if the user adds "+" to a task names we find all successor tasks in the graph and add them to the
        #  force execute list.
        tasks_to_force_execute = []
        for task_name in force:
            if task_name[-1] == '+':
                task_name = task_name[:-1]
                successor_tasks = [successor
                                   for successors in nx.dfs_successors(self.task_spec_graph, task_name).values()
                                   for successor in successors]
                tasks_to_force_execute.extend(successor_tasks)
            tasks_to_force_execute.append(task_name)
        tasks_to_force_execute = list(set(tasks_to_force_execute))

        # set force == True for all tasks in the created tasks_to_force_execute list
        for task in self._expanded_tasks:
            if task.name in tasks_to_force_execute:
                task.force = True

    def _order_task_specs(self,
                          task_specs: List[BaseTaskSpec],
                          ) -> List[BaseTaskSpec]:
        # task spec graph holding the user defined dependency structure
        task_spec_graph: DiGraph = self._create_task_spec_graph(task_specs=task_specs)

        # topological ordering of tasks in graph
        sorted_specs = [task_spec_graph.nodes[task_name]['task'] for task_name in topological_sort(task_spec_graph)]
        return sorted_specs

    @staticmethod
    def _validate_task_combination(task_combination: List[Task]) -> bool:
        def _match(task_cfgs: List[Dict[str, Any]]):
            # If the list of task_cfgs is empty we return True and continue with the next combination
            if not task_cfgs:
                return True

            unique_cfgs = []
            for config in task_cfgs:
                if config not in unique_cfgs:
                    unique_cfgs.append(config)

            # the predecessor tasks in the combination have no contradicting configs (they come from the same sweep)
            if len(unique_cfgs) == 1:
                return True
            return False

        # we validate the task combinations based on their path in the task graph
        # if two tasks have same parent task and their configs are different
        # then the combination is not valid
        task_names_in_path = list(
            set(name for task in task_combination for name in task.unique_config))

        # for each defined task name in path config
        for name in task_names_in_path:
            # we collect configs for this task name from each predecessor task in the task combination list
            # if a predecessor task is of type reduce or its config doesn't contain the above task name we skip
            task_configs = [task.unique_config[name] for task in task_combination
                            if not task.reduce and name in task.unique_config.keys()]

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
        task_configs = [task.unique_config for combination in task_combinations for task in combination]
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
    def _generate_task_nodes(task_specs: List[BaseTaskSpec]) -> List[Task]:
        # keep track of expanded tasks by their names
        expanded_tasks_by_name = defaultdict(list)
        task_id = 0

        # for each task to expand
        for exp_task in task_specs:
            # get predecessor task combinations
            task_combinations = Flow._get_predecessor_product(expanded_tasks_by_name, exp_task)

            if exp_task.reduce:
                # if it is a reduce task, just add the predecessor task
                # combinations as parents
                tasks = exp_task.build()

                for task in tasks:
                    # predecessor config
                    predecessor_config = Flow._merge_task_combination_configs(task_combinations, task_specs)

                    # add dependencies
                    for task_combination in task_combinations:
                        task.requires(task_combination)

                    task.id_ = task_id
                    task.reduce = True
                    expanded_tasks_by_name[task.name].append(task)
                    task.unique_config = {**predecessor_config, **{task.name: task.config_kwargs}}
                    task_id += 1
            else:
                # for each combination, create a new task
                for task_combination in task_combinations:
                    tasks = exp_task.build()

                    # shared predecessor config
                    predecessor_config = Flow._combine_task_config(task_combination)

                    # for each task that is created, add ids and dependencies
                    for task in tasks:
                        task.id_ = task_id
                        task.requires(task_combination)
                        task.unique_config = {**predecessor_config, **{task.name: task.config_kwargs}}
                        expanded_tasks_by_name[task.name].append(task)
                        task_id += 1

        # create final list of linked tasks and set expansion id for expanded tasks
        tasks = []
        for expanded_tasks in expanded_tasks_by_name.values():
            if len(expanded_tasks) == 1:
                for task in expanded_tasks:
                    tasks.append(task)
            else:
                for expansion_id, task in enumerate(expanded_tasks, 1):
                    task.unique_name = f'{task.name}-{expansion_id}'
                    tasks.append(task)

        return tasks

    @staticmethod
    def visualize(graph: DiGraph, use_ascii: Optional[bool] = None):
        """Visualizes the task graph by rendering it to the console via a pager
        -> keyboard input ":q" required to continue.

        Args:
            graph (DiGraph): a networkx directed graph object
            use_ascii (bool): renders the graph in ascii
                if None or False, renders in unicode if console supports it
        """

        console_graph = f'{graph.name}\n\n' if graph.name else ''
        console_graph += f'{create_console_graph(graph=graph, use_ascii=use_ascii)}\n\n'

        pager = FluidPager()
        pager.show(content=console_graph)

    def create(self,
               task_specs: List[BaseTaskSpec]):
        """Creates the task graph by expanding all GridTaskSpecs and taking reduce=True tasks into account.

        Args:
            task_specs (List[Union[TaskSpec, GridTaskSpec]]): list of task specifications
        """

        if not task_specs:
            raise NoTasksError("There are no tasks to run")

        Flow._check_no_task_name_clash(task_specs=task_specs)

        ordered_task_specs = self._order_task_specs(task_specs=task_specs)
        self._expanded_tasks: List[Task] = Flow._generate_task_nodes(ordered_task_specs)
        self.task_graph: DiGraph = Flow._create_graph_from_task_list(tasks=self._expanded_tasks, name='task graph')

    def run(self,
            force: Optional[Union[str, List[str]]] = None):
        """Runs the specified tasks and returns the results

        Args:
            force (Optional[str], optional): forcefully re-run tasks
                Possible options are:
                   1)  "all" - All the tasks are re-run
                   2)  a task name (eg. "PreProcessTask")
                       or list of task names (eg. ["PreProcessTask1", "PreProcessTask2])
                       Additionally, each task name can have the suffix '+' to re-run also its successors
                       (eg. "PreProcessTask+")

        Returns:
            Dict[str, Dict[str, Any]]: a nested dict of results
        """
        if self._expanded_tasks is None:
            raise NoTasksError('Execute "flow.create(tasks)" to build the task graph before calling "flow.run()".')

        if force is not None:
            self._register_tasks_to_force_execute(force=force)

        results = self._swarm.work(tasks=self._expanded_tasks)
        return results
