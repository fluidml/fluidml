from typing import List, Dict

import pytest

from fluidml.common import Resource
from fluidml.flow import Flow
from fluidml.flow import GridTaskSpec, TaskSpec
from fluidml.swarm import Swarm


@pytest.fixture
def parse(results: Dict, resource: Resource, in_dir: str):
    return {}


@pytest.fixture
def preprocess(results: Dict, resource: Resource, pipeline: List[str], abc: int):
    return {}


@pytest.fixture
def featurize_tokens(results: Dict, resource: Resource, type_: str, batch_size: int):
    return {}


@pytest.fixture
def featurize_cells(results: Dict, resource: Resource, type_: str, batch_size: int):
    return {}


@pytest.fixture
def train(results: Dict, resource: Resource, model, dataloader, evaluator, optimizer, num_epochs):
    return {'score': 2.}  # 'score': 2.


@pytest.fixture
def evaluate(results: Dict, resource: Resource, metric: str):
    print(results)
    return {}


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


def test_pipeline(dummy_resource):
    num_workers = 4

    # initialize all task specs
    parse_task = GridTaskSpec(task=parse, gs_config={"in_dir": "/some/dir"})
    preprocess_task = GridTaskSpec(task=preprocess, gs_config={"pipeline": ['a', 'b'], "abc": 1})
    featurize_tokens_task = GridTaskSpec(task=featurize_tokens, gs_config={"type_": "flair", 'batch_size': [2, 3]})
    featurize_cells_task = GridTaskSpec(task=featurize_cells, gs_config={"type_": "glove", "batch_size": 4})
    train_task = GridTaskSpec(task=train, gs_config={"model": "mlp", "dataloader": "x", "evaluator": "y", "optimizer": "adam", "num_epochs": 10})
    evaluate_task = TaskSpec(task=evaluate, reduce=True, task_kwargs={"metric": "accuracy"})

    # dependencies between tasks
    preprocess_task.requires([parse_task])
    featurize_tokens_task.requires([preprocess_task])
    featurize_cells_task.requires([preprocess_task])
    train_task.requires([featurize_tokens_task, featurize_cells_task])
    evaluate_task.requires([train_task])

    # pack all tasks in a list
    tasks = [parse_task,
             preprocess_task,
             featurize_tokens_task,
             featurize_cells_task,
             train_task,
             evaluate_task]

    # devices = get_balanced_devices(count=num_workers, use_cuda=True)
    resources = [dummy_resource(seed=42) for i in range(num_workers)]

    # create local file storage used for versioning, default InMemoryStore

    # run tasks in parallel (GridTaskSpecs are expanded based on grid search arguments)
    with Swarm(n_dolphins=num_workers,
               resources=resources,
               results_store=None) as swarm:
        flow = Flow(swarm=swarm, task_to_execute='evaluate', force=None)
        results = flow.run(tasks)
        assert len(results) == 14
