import pytest

from fluidml.flow import Flow
from fluidml.swarm import Swarm
from fluidml.common.exception import NoTasksError


@pytest.fixture
def swarm() -> Swarm:
    swarm = Swarm(n_dolphins=4)
    yield swarm
    swarm.close()


def test_flow_with_no_tasks(swarm):
    with pytest.raises(NoTasksError):
        flow = Flow(swarm=swarm)
        flow.run([])