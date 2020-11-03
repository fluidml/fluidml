import pytest

from fluidml.flow import Flow
from fluidml.swarm import Swarm
from fluidml.common.exception import NoTasksError
from fluidml.flow.task_spec import TaskSpec
# from fluidml.common.task import Task


@pytest.fixture
def swarm() -> Swarm:
    swarm = Swarm(n_dolphins=4)
    yield swarm
    swarm.close()


def test_flow_with_no_tasks(swarm):
    with pytest.raises(NoTasksError):
        flow = Flow(swarm=swarm)
        flow.run([])


# class DummyTask(Task):
#     def __init__(self, name: str, id_: int, x: int):
#         super().__init__(name=name, id_=id_)
#
#     def run(self, results, resource):
#         return {}


""" def test_flow_with_lambda(swarm):
    task_spec_a = TaskSpec(name="a", task=lambda results, resource, x: {}, task_kwargs={"x": 1})
    task_spec_b = TaskSpec(name="b", task=lambda results, resource, x: {}, task_kwargs={"x": 1})
    task_spec_b.requires([task_spec_a])
    flow = Flow(swarm)
    results = flow.run([task_spec_a, task_spec_b])
    assert len(results) == 2 """


def test_flow_with_dummy(swarm, dummy_task):
    task_spec_a = TaskSpec(name="a", task=dummy_task, task_kwargs={"x": 1})
    task_spec_b = TaskSpec(name="b", task=dummy_task, task_kwargs={"x": 1})
    task_spec_b.requires([task_spec_a])
    flow = Flow(swarm)
    results = flow.run([task_spec_a, task_spec_b])
    assert len(results) == 2
