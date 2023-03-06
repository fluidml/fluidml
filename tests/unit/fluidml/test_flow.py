import pytest

from fluidml.exception import NoTasksError
from fluidml.flow import Flow
from fluidml.swarm import Swarm
from fluidml.task_spec import TaskSpec


@pytest.fixture
def swarm() -> Swarm:
    swarm = Swarm(n_dolphins=4)
    yield swarm


def test_flow_with_no_tasks():
    with pytest.raises(NoTasksError):
        flow = Flow([])


def test_flow_with_dummy(dummy_task_a, dummy_task_b):
    task_spec_a = TaskSpec(name="A", task=dummy_task_a, config={"x": 1})
    task_spec_b = TaskSpec(name="B", task=dummy_task_b, config={"x": 1})
    task_spec_b.requires(task_spec_a)
    flow = Flow(tasks=[task_spec_a, task_spec_b])
    results = flow.run(return_results="all")
    assert len(results) == 2
