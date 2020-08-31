from busy_bee import Swarm, Task
import torch
import random


class SimpleModule(torch.nn.Module):
    def __init__(self, input_: int, output: int):
        super().__init__()
        self.linear = torch.nn.Linear(input_, output)

    def forward(self, input_: torch.Tensor):
        return self.linear(input_)


class TrainModuleTask(Task):
    def __init__(self, id_: int, n_inputs: int, n_outputs: int, epochs: int, batch_size: int, lr: float, device: str):
        super().__init__(id_)
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.model = SimpleModule(n_inputs, n_outputs).to(self.device)

    def run(self):
        optimizer = torch.optim.SGD(lr=self.lr,
                                    params=self.model.parameters())
        loss_criterion = torch.nn.MSELoss()
        epoch_losses = []
        for i in range(self.epochs):
            optimizer.zero_grad()
            inputs = torch.rand(self.batch_size, self.n_inputs).to(self.device)
            outputs = torch.rand(self.batch_size, self.n_outputs).to(self.device)
            preds = self.model.forward(inputs)
            loss = loss_criterion(preds, outputs)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.detach().cpu().numpy())
        return epoch_losses


def main():
    n_tasks = 10
    available_device = ["cpu"] if not torch.cuda.is_available() else [f"cuda:{i}" for i in
                                                                      range(torch.cuda.device_count())]
    devices = random.choices(available_device, k=n_tasks)

    tasks = {i: TrainModuleTask(i + 1, 10, 10, int(1e+4), 5, 1.0, devices[i]) for i in range(n_tasks)}

    with Swarm(graph=graph, n_bees=3, refresh_every=5) as swarm:
        results = swarm.work()
    print(results)


if __name__ == "__main__":
    main()
