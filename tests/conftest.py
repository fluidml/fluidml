from dataclasses import dataclass
from typing import Dict

import pytest

from fluidml.common import Task, Resource


class DummyTask(Task):
    def __init__(self, x: int):
        super().__init__()

    def run(self):
        self.results_store.save(obj={'a': 1}, name='a', type_='json')


@pytest.fixture
def dummy_task():
    return DummyTask


@dataclass
class TaskResource(Resource):
    seed: int


@pytest.fixture
def dummy_resource():
    return TaskResource
