from typing import List, Dict

from fluidml.common import Task
from fluidml.flow import Flow
from fluidml.flow import GridTaskSpec, TaskSpec
from fluidml.swarm import Swarm
from fluidml.storage import Sweep


class Parsing(Task):
    publishes = ["res1"]

    def __init__(self, in_dir: str, z: int):
        super().__init__()

        self.in_dir = in_dir
        self.z = z

    def run(self):
        self.save(obj={}, name="res1")


def preprocess(res1: Dict, pipeline: List[str], abc: List[int], task: Task):
    task.save(obj={}, name="res2", type_="pickle")


def featurize_tokens(res2: Dict, type_: str, batch_size: int, task: Task):
    task.save(obj={}, name="res3", type_="pickle")


def featurize_cells(res2: Dict, type_: str, batch_size: int, task: Task):
    task.save(obj={}, name="res4", type_="pickle")


def train(res3: Dict, res4: Dict, model, dataloader, evaluator, optimizer, num_epochs, task: Task):
    task.save(obj={}, name="res5", type_="pickle")


def evaluate(res5: List[Sweep], metric: str, task: Task):
    task.save(obj={}, name="res6", type_="pickle")


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
    parse_task = GridTaskSpec(task=Parsing, gs_config={"in_dir": "/some/dir"}, additional_kwargs={"z": 1})
    preprocess_task = GridTaskSpec(task=preprocess, gs_config={"pipeline": ["a", "b"], "abc": 1}, publishes=["res2"])
    featurize_tokens_task = GridTaskSpec(
        task=featurize_tokens, gs_config={"type_": "flair", "batch_size": [2, 3]}, publishes=["res3"]
    )
    featurize_cells_task = GridTaskSpec(
        task=featurize_cells, gs_config={"type_": "glove", "batch_size": 4}, publishes=["res4"]
    )
    train_task = GridTaskSpec(
        task=train,
        gs_config={"model": "mlp", "dataloader": "x", "evaluator": "y", "optimizer": "adam", "num_epochs": 10},
        publishes=["res5"],
    )
    evaluate_task = TaskSpec(task=evaluate, reduce=True, config={"metric": "accuracy"}, expects=["res5"])

    # dependencies between tasks
    preprocess_task.requires(parse_task)
    featurize_tokens_task.requires(preprocess_task)
    featurize_cells_task.requires(preprocess_task)
    train_task.requires([featurize_tokens_task, featurize_cells_task])
    evaluate_task.requires(train_task)

    # pack all tasks in a list
    tasks = [parse_task, preprocess_task, featurize_tokens_task, featurize_cells_task, train_task, evaluate_task]

    # devices = get_balanced_devices(count=num_workers, use_cuda=True)
    resources = [dummy_resource(device="cpu") for i in range(num_workers)]

    flow = Flow()
    flow.create(task_specs=tasks)

    # run tasks in parallel (GridTaskSpecs are expanded based on grid search arguments)
    with Swarm(n_dolphins=num_workers, resources=resources) as swarm:

        results = flow.run(swarm=swarm, force=None, results_store=None)

        num_expanded_tasks = 0
        for name, gs_runs in results.items():
            if isinstance(gs_runs, List):
                for run in gs_runs:
                    num_expanded_tasks += 1
            else:
                num_expanded_tasks += 1

        assert num_expanded_tasks == 14
