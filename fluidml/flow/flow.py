from collections import defaultdict
from itertools import product
from typing import List, Any, Dict, Optional

from networkx import DiGraph, shortest_path_length
from networkx.algorithms.dag import topological_sort

from fluidml.common import Task
from fluidml.flow.task_spec import BaseTaskSpec
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

    def _order_task_specs(self,
                          task_specs: List[BaseTaskSpec]) -> List[BaseTaskSpec]:
        # convert task_specs to dict, so it can be queried by name
        name_to_task_spec = {spec.name: spec for spec in task_specs}

        # task spec graph holding the user defined dependency structure
        task_spec_graph = DiGraph()
        for spec in task_specs:
            for predecessor in spec.predecessors:
                task_spec_graph.add_edge(predecessor.name, spec.name)

        # topological ordering of tasks in graph
        # if a specific task to execute is provided, remove non dependent tasks from graph
        if self._task_to_execute:
            sorted_names = list(shortest_path_length(task_spec_graph, target=self._task_to_execute).keys())[::-1]
        else:
            sorted_names = list(topological_sort(task_spec_graph))

        # get sorted list of task specs
        sorted_specs = [name_to_task_spec[name] for name in sorted_names]
        return sorted_specs

    @staticmethod
    def _get_predecessor_product(expanded_tasks_by_name: Dict[str, List[Task]],
                                 task_spec: BaseTaskSpec) -> List[List[Task]]:
        predecessor_tasks = [expanded_tasks_by_name[predecessor.name] for predecessor in task_spec.predecessors]
        task_combinations = [list(item) for item in product(*predecessor_tasks)] if predecessor_tasks else [[]]
        return task_combinations

    @staticmethod
    def _generate_tasks(task_specs: List[BaseTaskSpec]) -> List[Task]:
        # keep track of expanded tasks by their names
        expanded_tasks_by_name = defaultdict(list)
        task_id = 0

        # for each task to expand
        for exp_task in task_specs:
            # get predecessor task combinations
            task_combinations = Flow._get_predecessor_product(expanded_tasks_by_name, exp_task)

            # for each combination, create a new task
            for task_combination in task_combinations:
                tasks = exp_task.build()

                # for each task that is created, add ids and dependencies
                for task in tasks:
                    task.id_ = task_id
                    task.requires(task_combination)
                    expanded_tasks_by_name[task.name].append(task)
                    task_id += 1

        # create final dict of tasks
        tasks = [task for expanded_tasks in expanded_tasks_by_name.values() for task in expanded_tasks]
        return tasks

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
