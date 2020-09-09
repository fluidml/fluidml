from typing import List

import yaml
import os

from busy_bee.flow import Flow
from busy_bee.flow import GridTaskSpec
from busy_bee.hive import Swarm


def parse(in_dir: str):
    return {}


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

# task pipeline once grid search args are expanded
a1 -> b1 -> c1/d1 -> e1
                  -> e2
         -> c2/d1 -> e3
                  -> e4
"""


def main():
    current_dir = os.path.dirname(os.path.realpath(__file__))

    # load pipeline and config (tasks are named equally)
    pipeline = yaml.safe_load(open(os.path.join(current_dir, 'pipeline.yaml'), 'r'))
    config = yaml.safe_load(open(os.path.join(current_dir, 'config.yaml'), 'r'))

    # create task-spec objects and register dependencies (defined in pipeline.yaml)
    tasks = {task: GridTaskSpec(task=TASK_TO_CALLABLE[task], gs_config=config[task]) for task in pipeline}
    for task_name, task in tasks.items():
        task.requires([tasks[dep] for dep in pipeline[task_name]])

    # create list of task specs
    tasks = [task for task in tasks.values()]

    # run tasks in parallel (GridTaskSpecs are expanded based on grid search arguments)
    with Swarm(n_bees=2) as swarm:
        flow = Flow(swarm=swarm, task_to_execute='train')
        results = flow.run(tasks)

    print(results)


if __name__ == '__main__':
    main()
