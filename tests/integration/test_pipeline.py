import pathlib
from typing import Dict, List, Optional

from fluidml import Flow, Task, TaskSpec
from fluidml.storage import LocalFileStore, Sweep


class Parsing(Task):
    def __init__(self, in_dir: str, z: int):
        super().__init__()

        self.in_dir = in_dir
        self.z = z

    def run(self):
        self.save(obj={}, name="res1", type_="json")


def preprocess(res1: Dict, pipeline: List[str], abc: List[int], task: Task):
    task.save(obj={}, name="res2", type_="pickle")


def featurize_tokens(res2: Dict, type_: str, batch_size: int, task: Task):
    pass
    # task.save(obj={}, name="res3", type_="pickle")


def featurize_cells(res2: Dict, type_: str, batch_size: int, task: Task):
    task.save(obj={}, name="res4", type_="pickle")


def train(
    res4: Dict,
    model,
    dataloader,
    evaluator,
    optimizer,
    num_epochs,
    task: Task,
    res3: Optional[Dict] = None,
):
    task.save(obj={}, name="res5", type_="pickle")


def evaluate(res5: List[Sweep], metric: str, task: Task):
    task.save(obj={}, name="res6", type_="pickle")


# """
# # task pipeline
# parse -> preprocess -> featurize_tokens -> train -> evaluate
#                     \- featurize_cells  -/
#
# # task pipeline once grid search specs are expanded
# parse --> preprocess_1 -> featurize_tokens_1 ----> train -> evaluate (reduce grid search)
#                        \- featurize_tokens_2 --\/
#                        \- featurize_cells    --/\> train -/
#
#       \-> preprocess_2 -> featurize_tokens_1 ----> train -/
#                        \- featurize_tokens_2 --\/
#                        \- featurize_cells    --/\> train -/
# """


def test_pipeline(dummy_resource, tmp_path: pathlib.Path):
    num_workers = 4

    # initialize all task specs
    parse_task = TaskSpec(task=Parsing, config={"in_dir": "/some/dir"}, additional_kwargs={"z": 1})
    preprocess_task = TaskSpec(task=preprocess, config={"pipeline": ["a", "b"], "abc": 1}, expand="product")
    featurize_tokens_task = TaskSpec(
        task=featurize_tokens,
        config={"type_": "flair", "batch_size": [2, 3]},
        expand="product",
    )
    featurize_cells_task = TaskSpec(task=featurize_cells, config={"type_": "glove", "batch_size": 4})
    train_task = TaskSpec(
        task=train,
        config={
            "model": "mlp",
            "dataloader": "x",
            "evaluator": "y",
            "optimizer": "adam",
            "num_epochs": 10,
        },
    )
    evaluate_task = TaskSpec(task=evaluate, reduce=True, config={"metric": "accuracy"})

    # dependencies between tasks
    preprocess_task.requires(parse_task)
    featurize_tokens_task.requires(preprocess_task)
    featurize_cells_task.requires(preprocess_task)
    train_task.requires([featurize_tokens_task, featurize_cells_task])
    evaluate_task.requires(train_task)

    # pack all tasks in a list
    tasks = [
        parse_task,
        preprocess_task,
        featurize_tokens_task,
        featurize_cells_task,
        train_task,
        evaluate_task,
    ]

    # devices = get_balanced_devices(count=num_workers, use_cuda=True)
    resources = [dummy_resource(device="cpu") for i in range(num_workers)]

    # create local file storage used for versioning
    results_store = LocalFileStore(base_dir=str(tmp_path.resolve()))

    # create flow -> expand task graphs -> execute graph
    flow = Flow(tasks)
    results = flow.run(
        num_workers=num_workers,
        resources=resources,
        results_store=results_store,
        return_results="all",
    )

    num_expanded_tasks = 0
    for name, gs_runs in results.items():
        if isinstance(gs_runs, List):
            for run in gs_runs:
                num_expanded_tasks += 1
        else:
            num_expanded_tasks += 1
    assert num_expanded_tasks == 14
