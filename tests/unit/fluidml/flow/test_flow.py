import pytest

from fluidml.flow import Flow
from fluidml.swarm import Swarm
from fluidml.common.exception import NoTasksError
from fluidml.flow.task_spec import TaskSpec


@pytest.fixture
def swarm() -> Swarm:
    swarm = Swarm(n_dolphins=4)
    yield swarm


def test_flow_create_with_no_tasks(swarm):
    with pytest.raises(NoTasksError):
        flow = Flow(swarm=swarm)
        flow.create([])
        swarm.close()


def test_flow_run_with_no_tasks(swarm):
    with pytest.raises(NoTasksError):
        flow = Flow(swarm=swarm)
        flow.run([])
        swarm.close()


def test_flow_with_dummy(swarm, dummy_task_a, dummy_task_b):
    task_spec_a = TaskSpec(name="A", task=dummy_task_a, config={"x": 1}, publishes=['a'])
    task_spec_b = TaskSpec(name="B", task=dummy_task_b, config={"x": 1}, publishes=[])
    task_spec_b.requires(task_spec_a)
    flow = Flow(swarm=swarm)
    flow.create(task_specs=[task_spec_a, task_spec_b])
    results = flow.run()
    swarm.close()
    assert len(results) == 2
