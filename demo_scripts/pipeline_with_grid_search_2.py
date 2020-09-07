from typing import Callable, Dict, List, Optional, Any

import networkx as nx
import yaml

from busy_bee import Swarm, Task, Resource
from demo_scripts.utils import create_run_configs


def parse(in_dir: str):
    return {}
    # a - 3


def preprocess(pipeline: List[str]):
    return {}


def featurize_tokens(type_: str, batch_size: int):
    return {}


def featurize_cells(type_: str, batch_size: int):
    return {}


def train(model, dataloader, evaluator, optimizer, num_epochs):
    return {}


TASK_TO_CALLABLE = {'parse': parse,
                    'preprocess': preprocess,
                    'featurize_tokens': featurize_tokens,
                    'featurize_cells': featurize_cells,
                    'train': train}


"""
# task pipeline
a -> b -> c -> e
       \- d -/

# task pipeline including grid search args
a1 -> b1 -> c1/d1 -> e1
                  -> e2
         -> c2/d1 -> e3
                  -> e4
"""


class MyTask(Task):
    def __init__(self,
                 name: str,
                 task: Callable,
                 kwargs: Dict,
                 gs_config: Optional[Dict] = None):
        super().__init__(name=name)
        self.task = task
        self.kwargs = kwargs

    def run(self, results: Dict[str, Any], resource: Resource):
        result = self.task(**self.kwargs)
        return result


def get_entry_point_tasks(graph):
    tasks = []
    for node in graph.nodes:
        if len(list(graph.predecessors(node))) == 0:
            tasks.append(node)
    assert len(tasks) > 0, 'The dependency graph does not have any entry-point nodes.'
    return tasks


def main():
    # load pipeline and config (tasks are named equally)
    pipeline = yaml.safe_load(open('pipeline.yaml', 'r'))
    config = yaml.safe_load(open('config.yaml', 'r'))

    # split grid search config in individual run configs
    single_run_configs = create_run_configs(config_grid_search=config)

    # get all unique tasks, taking task dependency and kwargs dependency into account
    id_to_task_dict = {}
    task_id = 0
    for config in single_run_configs:
        for task_name, dependencies in pipeline.items():
            # create task object, holding its kwargs (config parameters)
            task = {'name': task_name,
                    'task': TASK_TO_CALLABLE[task_name],
                    'kwargs': config[task_name]}

            # add kwargs of dependent tasks for later comparison
            dep_kwargs = []
            for dep in dependencies:
                dep_kwargs.append(config[dep])
            task['dep_kwargs'] = dep_kwargs

            # check if task object is already included in task dictionary, skip if yes
            if task in id_to_task_dict.values():
                continue

            # if no, add task to task dictionary
            id_to_task_dict[task_id] = task

            # increment task_id counter
            task_id += 1

    # create graph object, to model task dependencies
    graph = nx.DiGraph()
    for id_i, task_i in id_to_task_dict.items():
        # for task_i, get kwargs of dependent tasks
        dep_kwargs = task_i['dep_kwargs']
        for id_j, task_j in id_to_task_dict.items():
            # for task_j, get kwargs
            kwargs = task_j['kwargs']
            # create an edge between task_j and task_i if task_j depends on task_i
            if kwargs in dep_kwargs:
                graph.add_edge(id_j, id_i)

    # convert task_dict to an abstract Task class -> Interface for swarm
    id_to_task = {}
    for id_, task_dict in id_to_task_dict.items():
        task = MyTask(id_=id_,
                      name=task_dict['name'],
                      task=task_dict['task'],
                      kwargs=task_dict['kwargs'])
        id_to_task[id_] = task

    # Register dependencies
    tasks = []
    for id_, task in id_to_task.items():
        pre_task_ids = list(graph.predecessors(id_))
        task.requires([id_to_task[id_] for id_ in pre_task_ids])
        tasks.append(task)

    # run swarm
    with Swarm(n_bees=3, refresh_every=5, exit_on_error=True) as swarm:
        results = swarm.work(tasks=tasks)


if __name__ == '__main__':
    main()
