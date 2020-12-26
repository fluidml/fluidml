from typing import Dict, List


def parse(results: Dict, in_dir: str):
    return {}


def preprocess(results: Dict, pipeline: List[str], abc: List[int]):
    return {}


def featurize_tokens(results: Dict, type_: str, batch_size: int):
    return {}


def featurize_cells(results: Dict, type_: str, batch_size: int):
    return {}


def train(results: Dict, model, dataloader, evaluator, optimizer, num_epochs):
    return {'score': 2.}  # 'score': 2.


def evaluate(results: Dict, metric: str):
    print(results)
    return {}
