from typing import Dict

import pytest

from fluidml.common.task import Task


class DummyTask(Task):
    def __init__(self, name: str, id_: int, x: int):
        super().__init__(name=name, id_=id_)

    def run(self, results, resource) -> Dict:
        return {}


@pytest.fixture
def dummy_task():
    return DummyTask
