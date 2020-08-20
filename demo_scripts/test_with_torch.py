from hive import Swarm, Task
import torch


class SimpleModule(torch.nn.Module):
    def __init__(self, input: int, output: int):
        super().__init__()
        self.linear = torch.nn.Linear(input, output)

    def forward(self, input: torch.Tensor):
        return self.linear(input)


class TrainModuleTask(Task):
    def __init__(self, id: int, n_inputs: int, n_outputs: int, epochs: int, batch_size: int, lr: float):
        super().__init__(id)
        self.model = SimpleModule(n_inputs, n_outputs)
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

    def run(self):
        optimizer = torch.optim.SGD(lr=self.lr, params=self.model.parameters())
        loss_criterion = torch.nn.MSELoss()
        epoch_losses = []
        for i in range(self.epochs):
            optimizer.zero_grad()
            inputs = torch.rand(self.batch_size, self.n_inputs)
            outputs = torch.rand(self.batch_size, self.n_outputs)
            preds = self.model.forward(inputs)
            loss = loss_criterion(preds, outputs)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.detach().numpy())
        return epoch_losses


if __name__ == "__main__":
    tasks = [TrainModuleTask(i+1, 10, 10, int(1e+4), 5, 1.0) for i in range(10)]
    swarm = Swarm(n_bees=3)
    results = swarm.work(tasks)
    swarm.close()
    print(results)