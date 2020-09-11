from argparse import ArgumentParser
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


CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


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


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--task',
                        default='train',
                        type=str,
                        help='Task to be executed (level 0 keys in config).')
    parser.add_argument('--config',
                        default=os.path.join(CURRENT_DIR, 'config.yaml'),
                        type=str,
                        help='Path to config file.',)
    parser.add_argument('--pipeline',
                        default=os.path.join(CURRENT_DIR, 'pipeline.yaml'),
                        type=str,
                        help='Path to pipeline file.',)
    parser.add_argument('--num-bees',
                        default=3,
                        type=int,
                        help='Number of spawned worker processes.')
    return parser.parse_args()


def main():
    args = parse_args()

    # load pipeline and config (tasks are named equally)
    pipeline = yaml.safe_load(open(args.pipeline, 'r'))
    config = yaml.safe_load(open(args.config, 'r'))

    # create task-spec objects and register dependencies (defined in pipeline.yaml)
    tasks = {task: GridTaskSpec(task=TASK_TO_CALLABLE[task], gs_config=config[task]) for task in pipeline}
    for task_name, task in tasks.items():
        task.requires([tasks[dep] for dep in pipeline[task_name]])

    # create list of task specs
    tasks = [task for task in tasks.values()]

    # run tasks in parallel (GridTaskSpecs are expanded based on grid search arguments)
    with Swarm(n_bees=args.num_bees) as swarm:
        flow = Flow(swarm=swarm, task_to_execute=args.task)
        results = flow.run(tasks)

    print(results)


if __name__ == '__main__':
    main()
