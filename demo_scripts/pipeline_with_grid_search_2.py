from typing import Callable, Dict, List, Optional, Any

import yaml
import os

from busy_bee.flow import Flow
from busy_bee.flow import GridTaskSpec
from busy_bee.hive import Swarm


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


def main():
    # load pipeline and config (tasks are named equally)
    current_dir = os.path.dirname(os.path.realpath(__file__))
    pipeline = yaml.safe_load(open(os.path.join(current_dir, 'pipeline.yaml'), 'r'))
    config = yaml.safe_load(open(os.path.join(current_dir, 'config.yaml'), 'r'))

    tasks = {task: GridTaskSpec(task=TASK_TO_CALLABLE[task], gs_config=config[task]) for task in pipeline}
    for task_name, task in tasks.items():
        task.requires([tasks[dep] for dep in pipeline[task_name]])

    tasks = [task for task in tasks.values()]

    with Swarm(n_bees=2) as swarm:
        flow = Flow(swarm=swarm)
        flow.run(tasks)


if __name__ == '__main__':
    main()
