import pytest

from fluidml.flow import Flow
from fluidml.swarm import Swarm


@pytest.fixture
def swarm() -> Swarm:
    swarm = Swarm(n_dolphins=4)
    yield swarm
    swarm.close()


def test_flow_instantiation(swarm):
    flow = Flow(swarm=swarm)
    flow.run([])
    assert True
