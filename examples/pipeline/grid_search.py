from argparse import ArgumentParser
from dataclasses import dataclass
import multiprocessing
import os
from typing import List, Dict, Optional

import torch
import yaml

from fluidml.common import Resource, Task
from fluidml.flow import Flow
from fluidml.flow import GridTaskSpec, TaskSpec
from fluidml.swarm import Swarm
from fluidml.storage import LocalFileStore, MongoDBStore, InMemoryStore


def get_balanced_devices(count: Optional[int] = None,
                         use_cuda: bool = True) -> List[str]:
    count = count if count is not None else multiprocessing.cpu_count()
    if use_cuda and torch.cuda.is_available():
        devices = [f'cuda:{id_}' for id_ in range(torch.cuda.device_count())]
    else:
        devices = ['cpu']
    factor = int(count / len(devices))
    remainder = count % len(devices)
    devices = devices * factor + devices[:remainder]
    return devices


def parse(in_dir: str, task: Task):
    task.results_store.save(obj=task.unique_config, name='config', type_='json',
                            task_name=task.name, task_unique_config=task.unique_config)
    task.results_store.save(obj={}, name='res1', type_='pickle',
                            task_name=task.name, task_unique_config=task.unique_config)


def preprocess(res1: Dict, pipeline: List[str], abc: List[int], task: Task):
    task.results_store.save(obj=task.unique_config, name='config', type_='json',
                            task_name=task.name, task_unique_config=task.unique_config)
    task.results_store.save(obj={}, name='res2', type_='pickle',
                            task_name=task.name, task_unique_config=task.unique_config)


def featurize_tokens(res2: Dict, type_: str, batch_size: int, task: Task):
    task.results_store.save(obj=task.unique_config, name='config', type_='json',
                            task_name=task.name, task_unique_config=task.unique_config)
    task.results_store.save(obj={}, name='res3', type_='pickle',
                            task_name=task.name, task_unique_config=task.unique_config)


def featurize_cells(res2: Dict, type_: str, batch_size: int, task: Task):
    task.results_store.save(obj=task.unique_config, name='config', type_='json',
                            task_name=task.name, task_unique_config=task.unique_config)
    task.results_store.save(obj={}, name='res4', type_='pickle',
                            task_name=task.name, task_unique_config=task.unique_config)


def train(res3: Dict, res4: Dict, model, dataloader, evaluator, optimizer, num_epochs, task: Task):
    task.results_store.save(obj=task.unique_config, name='config', type_='json',
                            task_name=task.name, task_unique_config=task.unique_config)
    task.results_store.save(obj={}, name='res5', type_='pickle',
                            task_name=task.name, task_unique_config=task.unique_config)


def evaluate(reduced_results: Dict, metric: str, task: Task):
    task.results_store.save(obj=task.unique_config, name='config', type_='json',
                            task_name=task.name, task_unique_config=task.unique_config)
    task.results_store.save(obj={}, name='res6', type_='pickle',
                            task_name=task.name, task_unique_config=task.unique_config)


CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


"""
# task pipeline
parse -> preprocess -> featurize_tokens -> train -> evaluate
                    \- featurize_cells  -/

# task pipeline once grid search specs are expanded
parse --> preprocess_1 -> featurize_tokens_1 ----> train -> evaluate (reduce grid search)
                       \- featurize_tokens_2 --\/
                       \- featurize_cells    --/\> train -/

      \-> preprocess_2 -> featurize_tokens_1 ----> train -/
                       \- featurize_tokens_2 --\/
                       \- featurize_cells    --/\> train -/
"""


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--task',
                        default='evaluate',
                        type=str,
                        help='Task to be executed (level 0 keys in config).')
    parser.add_argument('--force',
                        default='all',
                        choices=[None, 'all', 'selected'],
                        type=str,
                        help='Task to be executed (level 0 keys in config).')
    parser.add_argument('--use-cuda',
                        default=True,
                        help='If set, cuda (gpu) is used.',
                        action='store_true')
    parser.add_argument('--seed',
                        default=42,
                        type=int)
    parser.add_argument('--base-dir',
                        default=os.path.join(CURRENT_DIR, 'experiments'),
                        type=str)
    parser.add_argument('--config',
                        default=os.path.join(CURRENT_DIR, 'config.yaml'),
                        type=str,
                        help='Path to config file.',)
    parser.add_argument('--pipeline',
                        default=os.path.join(CURRENT_DIR, 'pipeline.yaml'),
                        type=str,
                        help='Path to pipeline file.',)
    parser.add_argument('--num-dolphins',
                        default=4,
                        type=int,
                        help='Number of spawned worker processes.')
    return parser.parse_args()


@dataclass
class TaskResource(Resource):
    device: str
    seed: int


def main():
    args = parse_args()

    # load pipeline and config (tasks are named equally)
    pipeline = yaml.safe_load(open(args.pipeline, 'r'))
    config = yaml.safe_load(open(args.config, 'r'))

    # create all task specs
    parse_task = GridTaskSpec(task=parse, publishes=['res1'], gs_config=config['parse'])
    preprocess_task = GridTaskSpec(task=preprocess, publishes=['res2'], gs_config=config['preprocess'])
    featurize_tokens_task = GridTaskSpec(task=featurize_tokens, publishes=['res3'], gs_config=config['featurize_tokens'])
    featurize_cells_task = GridTaskSpec(task=featurize_cells, publishes=['res4'], gs_config=config['featurize_cells'])
    train_task = GridTaskSpec(task=train, publishes=['res5'], gs_config=config['train'])
    evaluate_task = TaskSpec(task=evaluate, publishes=['res6'], task_kwargs=config['evaluate'], reduce=True)

    # register dependencies
    preprocess_task.requires([parse_task])
    featurize_tokens_task.requires([preprocess_task])
    featurize_cells_task.requires([preprocess_task])
    train_task.requires([featurize_tokens_task, featurize_cells_task])
    evaluate_task.requires([train_task])

    # create list of task specs
    tasks = [parse_task, preprocess_task, featurize_tokens_task, featurize_cells_task, train_task, evaluate_task]

    # create list of resources
    devices = get_balanced_devices(count=args.num_dolphins, use_cuda=args.use_cuda)
    resources = [TaskResource(device=devices[i],
                              seed=args.seed) for i in range(args.num_dolphins)]

    # create local file storage used for versioning
    results_store = LocalFileStore(base_dir=args.base_dir)
    # results_store = MongoDBStore("test3")

    # run tasks in parallel (GridTaskSpecs are expanded based on grid search arguments)
    with Swarm(n_dolphins=args.num_dolphins,
               resources=resources,
               results_store=results_store) as swarm:
        flow = Flow(swarm=swarm, task_to_execute=args.task, force=args.force)
        results = flow.run(tasks)
        print(results)


if __name__ == '__main__':
    main()


# def parse(results: Dict, in_dir: str):
#     return {}
#
#
# def preprocess(results: Dict, pipeline: List[str], abc: List[int]):
#     return {}
#
#
# def featurize_tokens(results: Dict, type_: str, batch_size: int):
#     return {}
#
#
# def featurize_cells(results: Dict, type_: str, batch_size: int):
#     return {}
#
#
# def train(results: Dict, model, dataloader, evaluator, optimizer, num_epochs):
#     return {'score': 2.}  # 'score': 2.
#
#
# def evaluate(results: Dict, metric: str):
#     print(results)
#     return {}
#
#
# CURRENT_DIR = os.getcwd()
# base_dir = os.path.join(CURRENT_DIR, 'experiments')
#
# parse_task = GridTaskSpec(task=parse, gs_config={"in_dir": "/some/dir"})
# preprocess_task = GridTaskSpec(task=preprocess, gs_config={"pipeline": ['a', 'b'], "abc": [1, 2]})
# featurize_tokens_task = GridTaskSpec(task=featurize_tokens, gs_config={"type_": "flair", 'batch_size': 2})
# featurize_cells_task = GridTaskSpec(task=featurize_cells, gs_config={"type_": "glove", "batch_size": 4})
# train_task = GridTaskSpec(task=train, gs_config={"model": "mlp", "dataloader": "x", "evaluator": "y", "optimizer": "adam", "num_epochs": 10})
# evaluate_task = TaskSpec(task=evaluate, reduce=True, task_kwargs={"metric": "accuracy"})
#
# # dependencies between tasks
# preprocess_task.requires([parse_task])
# featurize_tokens_task.requires([preprocess_task])
# featurize_cells_task.requires([preprocess_task])
# train_task.requires([featurize_tokens_task, featurize_cells_task])
# evaluate_task.requires([train_task])
#
# # all tasks
# tasks = [parse_task,
#          preprocess_task,
#          featurize_tokens_task,
#          featurize_cells_task,
#          train_task,
#          evaluate_task]
#
# results_store = LocalFileStore(base_dir=CURRENT_DIR)
#
# # run tasks in parallel (GridTaskSpecs are expanded based on grid search arguments)
# if __name__ == '__main__':
#     with Swarm(n_dolphins=4, results_store=results_store) as swarm:
#         flow = Flow(swarm=swarm)
#         results = flow.run(tasks)
#         print(results)
