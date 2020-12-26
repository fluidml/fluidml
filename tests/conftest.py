from dataclasses import dataclass
from typing import Dict

import pytest

from fluidml.common import Task, Resource


class DummyTask(Task):
    def __init__(self, name: str, id_: int, x: int):
        super().__init__(name=name, id_=id_)

    def run(self, results, resource) -> Dict:
        return {}


@pytest.fixture
def dummy_task():
    return DummyTask


@dataclass
class TaskResource(Resource):
    device: str
    seed: int


@pytest.fixture
def dummy_resource():
    return TaskResource
