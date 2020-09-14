from dataclasses import dataclass
from typing import Dict, Any

import torch

from busy_bee.hive import Swarm
from busy_bee.common import Task, Resource
from demo_scripts.utils.gpu import get_balanced_devices


class SimpleModule(torch.nn.Module):
    def __init__(self, input_: int, output: int):
        super().__init__()
        self.linear = torch.nn.Linear(input_, output)

    def forward(self, input_: torch.Tensor):
        return self.linear(input_)


@dataclass
class DeviceResource(Resource):
    device_id: str


class TrainModuleTask(Task):
    def __init__(self, id_: int, n_inputs: int, n_outputs: int, epochs: int, batch_size: int, lr: float):
        super().__init__(id_, "train_task")
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

    def run(self, results: Dict[str, Any], resource: Resource) -> Dict[str, Any]:
        device = resource.device_id
        model = SimpleModule(self.n_inputs, self.n_outputs)
        optimizer = torch.optim.SGD(lr=self.lr,
                                    params=model.parameters())
        loss_criterion = torch.nn.MSELoss()
        epoch_losses = []
        for i in range(self.epochs):
            optimizer.zero_grad()
            inputs = torch.rand(self.batch_size, self.n_inputs).to(device)
            outputs = torch.rand(self.batch_size, self.n_outputs).to(device)
            preds = model.forward(inputs)
            loss = loss_criterion(preds, outputs)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.detach().cpu().numpy())
        results = {"epoch_losses": epoch_losses}
        return results


def main():
    n_tasks = 10
    n_bees = 3

    resources = get_balanced_devices(count=n_bees, no_cuda=False)
    resources = [DeviceResource(device) for device in resources]

    tasks = [TrainModuleTask(i + 1, 10, 10, int(1e+4), 5, 1.0) for i in range(n_tasks)]

    with Swarm(n_bees=n_bees, refresh_every=5, resources=resources) as swarm:
        results = swarm.work(tasks)

    print(results[1])


if __name__ == "__main__":
    main()
