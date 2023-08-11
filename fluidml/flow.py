import logging
import multiprocessing
import sys
from collections import defaultdict
from itertools import product
from typing import Any, Dict, List, Optional, Set, Union

import networkx as nx
from metadict import MetaDict
from networkx import DiGraph
from networkx.algorithms.dag import topological_sort

from fluidml.exception import CyclicGraphError, NoTasksError, TaskNameError
from fluidml.storage import InMemoryStore, ResultsStore
from fluidml.swarm import Swarm
from fluidml.task import Task, TaskResults
from fluidml.task_spec import TaskSpec
from fluidml.utils import generate_run_name, reformat_config, update_merge

logger = logging.getLogger(__name__)


class Flow:
    """Flow implements the core logic of building and expanding task graphs from task specifications.

    * It automatically expands the tasks based on task specifications and task configs.
    * It extends the registered dependencies to the expanded tasks.
    * It provides the task graph objects for introspection and visualization.
    * Finally, it executes the task graph using multiprocessing if necessary.

    Args:
        tasks: List of task specifications.
        config_ignore_prefix: A config key prefix, e.g. "_". Prefixed keys will be not included in the
            "unique_config", which is used to determine whether a run has been executed or not.
        config_group_prefix: A config grouping prefix, to indicate that to parameters are grouped and expanded
            using the "zip" method. The grouping prefix enables the "zip" expansion of specific parameters, while
            all remaining grid parameters are expanded via "product".
            Example: ``cfg = {"a": [1, 2, "@x"], "b": [1, 2, 3], "c": [1, 2, "@x"]``
            Without grouping "product" expansion would yield: `2 * 2 * 3 = 12` configs.
            With grouping "product" expansion yields : `2 * 3 = 6` configs, since the grouped parameters are
            "zip" expanded.
    """

    def __init__(
        self,
        tasks: List[TaskSpec],
        config_ignore_prefix: Optional[str] = None,
        config_group_prefix: Optional[str] = None,
    ):
        # assign config_ignore_prefix and config_group_prefix to all tasks if the task has no prefix assigned, yet
        for task_spec in tasks:
            if config_group_prefix is not None and task_spec.config_group_prefix is None:
                task_spec.config_group_prefix = config_group_prefix
            if config_ignore_prefix is not None and task_spec.config_ignore_prefix is None:
                task_spec.config_ignore_prefix = config_ignore_prefix

        # contains the expanded graph as list of Task objects -> used internally in swarm
        self._expanded_tasks: Optional[List[Task]] = None
        # contains the expanded graph as networkx DiGraph -> accessible to the user for introspection and visualization
        self.task_graph: Optional[DiGraph] = None
        # contains the original, user defined task spec graph as networkx DiGraph
        # -> accessible to the user for introspection and visualization
        self.task_spec_graph: Optional[DiGraph] = None

        # create expanded graph and set above attributes
        self._create(task_specs=tasks)

    @property
    def num_tasks(self) -> int:
        """The total number of expanded tasks in the task graph."""
        return len(self._expanded_tasks)

    def _create(self, task_specs: List[TaskSpec]):
        if not task_specs:
            raise NoTasksError("There are no tasks to run.")

        Flow._check_no_task_name_clash(task_specs=task_specs)

        ordered_task_specs = self._order_task_specs(task_specs=task_specs)
        self._expanded_tasks: List[Task] = Flow._expand_and_link_tasks(ordered_task_specs)
        self.task_graph: DiGraph = Flow._create_graph_from_task_spec_list(
            task_specs=self._expanded_tasks, name="task graph"
        )

    def _infer_optimal_number_of_workers_from_graph(self) -> int:
        """Analyzes the graphs layout and finds the maximal number of tasks executed in parallel."""
        # get root tasks
        root_tasks = [task.unique_name for task in self._expanded_tasks if len(task.predecessors) == 0]

        # assign to each node the node's level in the directed graph
        node_to_lvl = {}
        for root in root_tasks:
            # assign lvl 0 to all root nodes
            lvl = 0
            node_to_lvl[root] = lvl
            # assign lvl to all successor nodes recursively
            Flow._assign_lvl_to_node_recursively(root, self.task_graph, lvl, node_to_lvl)

        # get the amount of nodes per lvl in the graph
        lvl_to_num = defaultdict(int)
        for node, lvl in node_to_lvl.items():
            lvl_to_num[lvl] += 1

        # return the maximum amount of nodes on the same level
        # if possible this equals the optimal amount of processes for parallelization
        optimal_num_workers = max([num for num in lvl_to_num.values()])
        return optimal_num_workers

    @staticmethod
    def _assign_lvl_to_node_recursively(node: str, graph: nx.DiGraph, lvl: int, node_to_lvl: Dict[str, int]):
        lvl += 1
        for successor in graph.successors(node):
            if successor not in node_to_lvl or node_to_lvl[successor] < lvl:
                node_to_lvl[successor] = lvl
            Flow._assign_lvl_to_node_recursively(successor, graph, lvl, node_to_lvl)

    @staticmethod
    def _check_acyclic(graph: DiGraph) -> None:

        # try to find cycles in graph -> return None if None are found
        try:
            edges = nx.find_cycle(graph, orientation="original")
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
    def _check_no_task_name_clash(task_specs: List[TaskSpec]) -> None:
        from importlib import import_module

        for task_spec in task_specs:
            task_obj = task_spec.task
            task_name = task_spec.task.__name__
            module_name = getattr(task_obj, "__module__")
            import_module(module_name)
            module = sys.modules[module_name]
            obj = getattr(module, task_name)
            if obj is not task_obj:
                raise TaskNameError(
                    f"Task names have to be unique. "
                    f'A second object different from task "{task_name}" was found with the same name: \n'
                    f"{obj} in {module}."
                )

    @staticmethod
    def _create_graph_from_task_spec_list(
        task_specs: List[Union[TaskSpec, Task]], name: Optional[str] = None
    ) -> DiGraph:
        """Creates nx.DiGraph object of the list of defined tasks with registered dependencies."""
        graph = DiGraph()
        for task_spec in task_specs:
            graph.add_node(task_spec.unique_name, task=task_spec)
            for predecessor in task_spec.predecessors:
                graph.add_node(predecessor.unique_name, task=predecessor)
                graph.add_edge(predecessor.unique_name, task_spec.unique_name)
        if name is not None:
            graph.name = name
        return graph

    def _create_task_spec_graph(self, task_specs: List[TaskSpec]) -> DiGraph:
        task_spec_graph = Flow._create_graph_from_task_spec_list(task_specs=task_specs, name="task spec graph")

        # assure that task spec graph contains no cyclic dependencies
        Flow._check_acyclic(graph=task_spec_graph)
        self.task_spec_graph = task_spec_graph
        return task_spec_graph

    def _register_tasks_to_force_execute(self, force: Union[str, List[str]]) -> None:
        # make sure that "all" is provided in the correct way, either as str or as str in a list of length 1
        if isinstance(force, List):
            if len(force) == 1 and force[0] == "all":
                force = force[0]
            elif "all" in force:
                raise TypeError('"all" must be provided as str or list of length 1 to the "force" argument.')

        # if force == 'all' set force to True for all tasks
        if force == "all":
            for task in self._expanded_tasks:
                task.force = True
            return

        # convert to list if force is of type str
        if isinstance(force, str):
            force = [force]

        if not isinstance(force, List):
            raise TypeError('"force" argument has to be of type str or list of str.')

        # get all user provided task names to force execute
        force_task_names = [task_name[:-1] if task_name[-1] == "+" else task_name for task_name in force]
        # get all task names defined in the task spec graph
        task_names = list(self.task_spec_graph.nodes)
        # find all user provided task names to force execute that don't exist in the task spec graph
        unknown_task_names = list(set(force_task_names).difference(task_names))
        # if any unknown names are found, raise a ValueError
        if unknown_task_names:
            raise ValueError(
                f'The following task names provided to "force" are unknown: ' f'{", ".join(unknown_task_names)}'
            )

        # create a list of unique task names to force execute
        # if the user adds "+" to a task names we find all successor tasks in the graph and add them to the
        #  force execute list.
        tasks_to_force_execute = []
        for task_name in force:
            if task_name[-1] == "+":
                task_name = task_name[:-1]
                successor_tasks = [
                    successor
                    for successors in nx.dfs_successors(self.task_spec_graph, task_name).values()
                    for successor in successors
                ]
                tasks_to_force_execute.extend(successor_tasks)
            tasks_to_force_execute.append(task_name)
        tasks_to_force_execute = list(set(tasks_to_force_execute))

        # set force == True for all tasks in the created tasks_to_force_execute list
        for task in self._expanded_tasks:
            if task.name in tasks_to_force_execute:
                task.force = True

    def _order_task_specs(
        self,
        task_specs: List[TaskSpec],
    ) -> List[TaskSpec]:
        # task spec graph holding the user defined dependency structure
        task_spec_graph: DiGraph = self._create_task_spec_graph(task_specs=task_specs)

        # topological ordering of tasks in graph
        sorted_specs = [task_spec_graph.nodes[task_name]["task"] for task_name in topological_sort(task_spec_graph)]
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
        task_names_in_path = list(set(name for task in task_combination for name in task.unique_config))

        # for each defined task name in path config
        for name in task_names_in_path:
            # we collect configs for this task name from each predecessor task in the task combination list
            # if a predecessor task is of type reduce or its config doesn't contain the above task name we skip
            task_configs = [
                task.unique_config[name]
                for task in task_combination
                if not task.reduce and name in task.unique_config.keys()
            ]

            # if they do not match, return False
            if not _match(task_configs):
                return False
        return True

    @staticmethod
    def _get_predecessor_product(expanded_tasks_by_name: Dict[str, List[Task]], spec: TaskSpec) -> List[List[Task]]:
        predecessor_tasks = [expanded_tasks_by_name[predecessor.name] for predecessor in spec.predecessors]
        task_combinations = [list(item) for item in product(*predecessor_tasks)] if predecessor_tasks else [[]]
        task_combinations = [
            combination for combination in task_combinations if Flow._validate_task_combination(combination)
        ]
        return task_combinations

    @staticmethod
    def _combine_task_config(task_combination: List[Task]) -> Dict:
        config = {}
        for task in task_combination:
            config = {**config, **task.unique_config}
        return config

    @staticmethod
    def _merge_task_combination_configs(task_combinations: List[List[Task]], task_specs: List[TaskSpec]) -> Dict:

        task_configs = [task.unique_config for combination in task_combinations for task in combination]
        merged_config = task_configs.pop(0)
        for config in task_configs:
            merged_config: Dict = update_merge(merged_config, config)

        if task_configs:
            # get all task names that were specified as GridTaskSpec
            grid_task_names = [spec.name for spec in task_specs if spec.expand_fn is not None]

            # split merged_config in grid_task_config and normal_task_config
            grid_task_config = {key: value for key, value in merged_config.items() if key in grid_task_names}
            normal_task_config = {key: value for key, value in merged_config.items() if key not in grid_task_names}

            # reformat only grid_task_config (replace tuples by lists)
            grid_task_config: Dict = reformat_config(grid_task_config)

            # merge back the normal_task_config with the formatted grid_task_config
            merged_config = {**normal_task_config, **grid_task_config}
        return merged_config

    @staticmethod
    def _expand_and_link_tasks(specs: List[TaskSpec]) -> List[Task]:
        # keep track of expanded task_specs by their names
        expanded_tasks_by_name = defaultdict(list)

        # for each spec to expand
        for spec in specs:
            # get predecessor task combinations
            task_combinations = Flow._get_predecessor_product(expanded_tasks_by_name, spec)

            if spec.reduce:
                # if it is a reduce task, just add the predecessor task combinations as parents
                expanded_tasks = spec.expand()

                for task in expanded_tasks:
                    # merge configs from all predecessors to get a single reduced predecessor config
                    # note: this config might differ from the original grid config since original grid lists
                    #  containing dicts can not be recovered when merging expanded configs
                    predecessor_config = Flow._merge_task_combination_configs(task_combinations, specs)

                    # add dependencies
                    for task_combination in task_combinations:
                        task.requires(task_combination)

                    # task.counter = task_counter
                    expanded_tasks_by_name[task.name].append(task)

                    task.unique_config = MetaDict(
                        {
                            **predecessor_config,
                            **{task.name: task.unique_config},
                        }
                    )
            else:
                # for each combination, create a new task
                for task_combination in task_combinations:
                    expanded_tasks = spec.expand()

                    # shared predecessor config
                    predecessor_config = Flow._combine_task_config(task_combination)

                    # for each task that is created, add ids and dependencies
                    for task in expanded_tasks:
                        task.requires(task_combination)

                        task.unique_config = MetaDict(
                            {
                                **predecessor_config,
                                **{task.name: task.unique_config},
                            }
                        )
                        expanded_tasks_by_name[task.name].append(task)

        # create final list of linked task specs and set expansion id for expanded specs
        all_tasks = []
        for expanded_tasks in expanded_tasks_by_name.values():
            if len(expanded_tasks) == 1:
                for task in expanded_tasks:
                    all_tasks.append(task)
            else:
                for expansion_id, task in enumerate(expanded_tasks, 1):
                    task.unique_name = f"{task.name}-{expansion_id}"
                    all_tasks.append(task)

        return all_tasks

    def _get_number_of_used_workers(self, num_workers: Optional[int] = None) -> int:
        """Get the number of workers, given the optimal, maximum and user-defined number.

        1. get maximum number of available workers.
        2. infer optimal number of workers given the expanded graph to process
        3. define num_workers based on optimal_num_workers

        Args:
            num_workers: Number of workers set by the user.

        Returns:
            Number of workers used.
        """
        max_num_workers = multiprocessing.cpu_count()
        optimal_num_workers = self._infer_optimal_number_of_workers_from_graph()
        optimal_num_workers = optimal_num_workers if optimal_num_workers <= max_num_workers else max_num_workers
        if num_workers is None or num_workers > optimal_num_workers:
            num_workers = optimal_num_workers

        return num_workers

    @staticmethod
    def _process_resources(num_workers: int, resources: Optional[Any] = None) -> Optional[List[Any]]:
        # convert resources to list if a single resource was provided
        # and trim list of resources to actual number of used workers
        if resources is None:
            return None
        elif isinstance(resources, List):
            return resources[:num_workers]
        else:
            return [resources]

    def run(
        self,
        num_workers: Optional[int] = None,
        resources: Optional[Union[Any, List[Any]]] = None,
        start_method: str = "spawn",
        exit_on_error: bool = True,
        log_to_tmux: bool = False,
        max_panes_per_window: int = 4,
        force: Optional[Union[str, List[str]]] = None,
        project_name: str = "uncategorized",
        run_name: Optional[str] = None,
        results_store: Optional[ResultsStore] = None,
        return_results: Optional[str] = None,
    ) -> Optional[Dict[str, List[TaskResults]]]:
        """Runs the expanded task graph sequentially or in parallel using multiprocessing and returns the results.

        Args:
            num_workers: Number of parallel processes (dolphins) used. Internally, the optimal number of processes
                ``optimal_num_workers`` is inferred from the task graph. If ``num_workers is None``
                or ``num_workers > optimal_num_workers``, ``num_workers`` is overwritten to ``optimal_num_workers``.
                Defaults to ``None``.
            resources: A single resource object or a list of resources that are assigned to workers. Resources can
                hold arbitrary data, e.g. gpu or cpu device information, making sure that each worker has access to a
                dedicated device
                If ``num_workers > 1`` and ``len(resources) == num_workers`` resources are assigned to each worker.
                If ``len(resources) < num_workers`` resources are assigned randomly to workers.
                If ``num_workers == 1`` the first resource ``resources[0]`` is assigned to all tasks (not workers).
                Defaults to ``None``.
            start_method: Start method for multiprocessing. Defaults to ``"spawn"``. Only used when ``num_workers > 1``.
            exit_on_error: When an error happens all workers finish their current tasks and exit gracefully.
                Defaults to True. Only used when ``num_workers > 1``.
            log_to_tmux: If ``True`` a new tmux session is created (given tmux is installed) and each worker (process)
                logs to a dedicated pane to avoid garbled logging output. The session's name equals the combined
                ``f"{project_name}--{run-name}"``. Defaults to ``False``. Only used when ``num_workers > 1``.
            max_panes_per_window: Max number of panes per tmux window. Requires ``log_to_tmux`` being set to ``True``.
                Defaults to ``4``. Only used when ``num_workers > 1``.
            force: Forcefully re-run tasks. Possible options are:
                1) ``"all"`` - All the tasks are re-run.
                2) A task name (e.g. "PreProcessTask")
                or list of task names (e.g. ``["PreProcessTask1", "PreProcessTask2"]``). Additionally, each task name
                can have the suffix "+" to re-run also its successors (e.g. "PreProcessTask+").
            project_name: Name of project. Defaults to ``"uncategorized"``.
            run_name: Name of run.
            results_store: An instance of results store for results management.
                If nothing is provided, a non-persistent InMemoryStore store is used.
            return_results: Return results-dictionary after ``run()``. Defaults to ``all``.
                Choices: ``"all", "latest", None``.

        Returns:
            A results dictionary with the following schema.
                ``{"task_name_1": List[TaskResults], "task_name_2": List[TaskResults]}``
        """

        if force is not None:
            self._register_tasks_to_force_execute(force=force)

        # generate random run name in the form of "adjective-noun"
        if run_name is None:
            run_name = generate_run_name()

        # if InMemoryStore is used, return the latest pipeline results
        if return_results is None and (results_store is None or isinstance(results_store, InMemoryStore)):
            return_results = "latest"

        # get maximum number of available workers
        # and infer optimal number of workers given the expanded graph to process
        num_workers = self._get_number_of_used_workers(num_workers)

        # convert resources to list if a single resource was provided
        # and trim list of resources to actual number of used workers
        resources = self._process_resources(num_workers, resources)

        with Swarm(
            n_dolphins=num_workers,
            resources=resources,
            start_method=start_method,
            exit_on_error=exit_on_error,
            log_to_tmux=log_to_tmux,
            max_panes_per_window=max_panes_per_window,
        ) as swarm:
            results = swarm.work(
                tasks=self._expanded_tasks,
                run_name=run_name,
                project_name=project_name,
                results_store=results_store,
                return_results=return_results,
            )

        return results
