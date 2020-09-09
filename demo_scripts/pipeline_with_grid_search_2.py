from typing import Callable, Dict, List, Optional, Any

import networkx as nx
import yaml

from busy_bee.flow import Flow
from busy_bee.flow import TaskSpec, GridTaskSpec


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

# def get_predecessor_tasks(list_of_tasks, task):
#     if task.predecessors:
#         for pred_task in task.predecessors:
#             list_of_tasks.append(pred_task)
#             return get_predecessor_tasks(list_of_tasks, pred_task)
#     else:
#         return list_of_tasks




def main():
    TASK_TO_EXECUTE = 'featurize_tokens'

    # load pipeline and config (tasks are named equally)
    pipeline = yaml.safe_load(open('pipeline.yaml', 'r'))
    config = yaml.safe_load(open('config.yaml', 'r'))

    tasks = {task_name: GridTaskSpec(task=TASK_TO_CALLABLE[task_name], gs_config=config[task_name])
             for task_name in pipeline}

    graph = nx.DiGraph()
    for task_name, task in tasks.items():
        task.requires([tasks[dep] for dep in pipeline[task_name]])
        for dep_name in pipeline[task_name]:
            graph.add_edge(dep_name, task_name)

    # list_of_tasks = get_predecessor_tasks(list_of_tasks=[tasks[TASK_TO_EXECUTE]], task=tasks[TASK_TO_EXECUTE])

    list_of_tasks = [tasks[name] for name in nx.ancestors(graph, TASK_TO_EXECUTE)] + [tasks[TASK_TO_EXECUTE]]

    flow = Flow(tasks=list_of_tasks)
    flow.run()


if __name__ == '__main__':
    main()
