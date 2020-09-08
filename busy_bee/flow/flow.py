from collections import defaultdict
from copy import deepcopy
from itertools import product
from types import FunctionType
from typing import List, Any, Dict, Union, Tuple

from busy_bee.common import Task, Resource
from busy_bee.hive import Swarm
from busy_bee.flow.task_spec import TaskSpec, GridTaskSpec


class MyTask(Task):
    def __init__(self,
                 id_: int,
                 name: str,
                 task: FunctionType,
                 kwargs: Dict):
        super().__init__(id_=id_, name=name)
        self.task = task
        self.kwargs = kwargs

    def run(self, results: Dict[str, Any], resource: Resource):
        result = self.task(**self.kwargs)
        return result


def _find_list_in_dict(obj: Dict, param_grid: List) -> List:
    for key in obj:
        if isinstance(obj[key], list):
            param_grid.append([val for val in obj[key]])
        elif isinstance(obj[key], dict):
            _find_list_in_dict(obj[key], param_grid)
        else:
            continue
    return param_grid

def _replace_list_in_dict(obj: Dict, obj_copy: Dict, comb: Tuple, counter: List) -> Tuple[Dict, List]:
    for key, key_copy in zip(obj, obj_copy):
        if isinstance(obj[key], list):
            obj_copy[key_copy] = comb[len(counter)]
            counter.append(1)
        elif isinstance(obj[key], dict):
            _replace_list_in_dict(obj[key], obj_copy[key_copy], comb, counter)
        else:
            continue
    return obj_copy, counter

def split_gs_config(config_grid_search: Dict) -> List[Dict]:
    param_grid = []
    param_grid = _find_list_in_dict(config_grid_search, param_grid)
    config_copy = deepcopy(config_grid_search)
    individual_configs = []
    for comb in product(*param_grid):
        counter = []
        individual_config = _replace_list_in_dict(config_grid_search, config_copy, comb, counter)[0]
        individual_config = deepcopy(individual_config)
        individual_configs.append(individual_config)
    return individual_configs


class Flow:
    def __init__(self,
                 tasks: List[Union[TaskSpec, GridTaskSpec]]):
        self._tasks = tasks

    def get_list_of_tasks(self):

        # we just have to convert tasks/task_names to graphs
        # and get the task expansion order
        tasks_to_expand = []
        tasks_to_expand = self._tasks

        # expand the tasks in the topological order
        expanded_tasks_by_name = defaultdict(list)
        task_id = 0
        # for each task to expand
        for exp_task in tasks_to_expand:
            # for each predecessor task
            if exp_task.predecessors:

                    # for predecessor task in expanded tasks of my predecessor
                    # for pred_task in expanded_tasks_by_name[predecessor.name]:
                exp_task_configs = exp_task.task_configs if isinstance(exp_task, GridTaskSpec) else [exp_task.task_kwargs]
                # expand the current task config
                for kwargs in exp_task_configs:
                    for predecessor in exp_task.predecessors:
                        if isinstance(exp_task.task, type):
                            task = exp_task.task(id_=task_id, name=exp_task.name, **kwargs)
                        elif isinstance(exp_task.task, FunctionType):
                            task = MyTask(id_=task_id, task=exp_task.task, name=exp_task.name, kwargs=kwargs)
                        else:
                            raise ValueError
                        task.requires(expanded_tasks_by_name[predecessor.name])
                        expanded_tasks_by_name[task.name].append(task)
                        task_id += 1
            else:
                task_configs = exp_task.task_configs if isinstance(exp_task, GridTaskSpec) else [exp_task.task_kwargs]
                # expand the current task config
                for kwargs in task_configs:
                    if isinstance(exp_task.task, type):
                        task = exp_task.task(id_=task_id, name=exp_task.name, **kwargs)
                    elif isinstance(exp_task.task, FunctionType):
                        task = MyTask(id_=task_id, task=exp_task.task, name=exp_task.name, kwargs=kwargs)
                    else:
                        raise ValueError
                    expanded_tasks_by_name[task.name].append(task)
                    task_id += 1
        tasks = [task for expanded_tasks in expanded_tasks_by_name.values() for task in expanded_tasks]

        for task in tasks:
            print('id', task.id_)
            print('name', task.name)
            for pred in task.predecessors:
                print('  id', pred.id_, pred.name)

        return tasks
    #
    # def get_list_of_tasks_new(self):
    #
    #     gs_config = {}
    #     for task in self._tasks:
    #         gs_config[task.name] = task.gs_config if isinstance(task, GridTaskSpec) else task.task_kwargs
    #
    #     single_run_configs = split_gs_config(config_grid_search=gs_config)
    #
    #
    #     id_to_task_dict = {}
    #     task_id = 0
    #     for exp_task in self._tasks:
    #         exp_task_configs = exp_task.task_configs if isinstance(exp_task, GridTaskSpec) else [exp_task.task_kwargs]
    #         for kwargs in exp_task_configs:
    #             task = {'name': exp_task.name,
    #                     'task': exp_task.task,
    #                     'kwargs': kwargs}
    #             if exp_task.predecessors:
    #                 dep_kwargs = []
    #                 for dep_task in exp_task.predecessors:
    #                     dep_task_configs = dep_task.task_configs if isinstance(dep_task, GridTaskSpec) else [
    #                         dep_task.task_kwargs]
    #
    #
    #
    #
    #     for config in single_run_configs:
    #         for task_name, dependencies in pipeline.items():
    #             # create task object, holding its kwargs (config parameters)
    #             task = {'name': task_name,
    #                     'task': TASK_TO_CALLABLE[task_name],
    #                     'kwargs': config[task_name]}
    #
    #             # add kwargs of dependent tasks for later comparison
    #             dep_kwargs = []
    #             for dep in dependencies:
    #                 dep_kwargs.append(config[dep])
    #             task['dep_kwargs'] = dep_kwargs
    #
    #             # check if task object is already included in task dictionary, skip if yes
    #             if task in id_to_task_dict.values():
    #                 continue
    #
    #             # if no, add task to task dictionary
    #             id_to_task_dict[task_id] = task
    #
    #             # increment task_id counter
    #             task_id += 1
    #
    #     # create graph object, to model task dependencies
    #     graph = nx.DiGraph()
    #     for id_i, task_i in id_to_task_dict.items():
    #         # for task_i, get kwargs of dependent tasks
    #         dep_kwargs = task_i['dep_kwargs']
    #         for id_j, task_j in id_to_task_dict.items():
    #             # for task_j, get kwargs
    #             kwargs = task_j['kwargs']
    #             # create an edge between task_j and task_i if task_j depends on task_i
    #             if kwargs in dep_kwargs:
    #                 graph.add_edge(id_j, id_i)
    #
    #     # convert task_dict to an abstract Task class -> Interface for swarm
    #     id_to_task = {}
    #     for id_, task_dict in id_to_task_dict.items():
    #         task = MyTask(id_=id_,
    #                       name=task_dict['name'],
    #                       task=task_dict['task'],
    #                       kwargs=task_dict['kwargs'])
    #         id_to_task[id_] = task
    #
    #     # Register dependencies
    #     tasks = []
    #     for id_, task in id_to_task.items():
    #         pre_task_ids = list(graph.predecessors(id_))
    #         task.requires([id_to_task[id_] for id_ in pre_task_ids])
    #         tasks.append(task)

    def run(self) -> Dict[str, Dict[str, Any]]:
        """
        Runs the provided tasks

        Args:

        Returns:
            Dict[str, Dict[str, any]] - a nested dictionary of results
        """

        tasks = self.get_list_of_tasks()

        # run swarm
        with Swarm(n_bees=3, refresh_every=5, exit_on_error=True) as swarm:
            results = swarm.work(tasks=tasks)

        return results

        # 1. first expand the tasks that are grid searcheable

        # 2. also, take care of their dependencies

        # 3. get a final list of tasks

        # 4. run the tasks through swarm

        # 5. return results
