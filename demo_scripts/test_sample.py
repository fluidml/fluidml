from hive import Swarm, Task


class SampleTask(Task):
    def __init__(self, id: int):
        super().__init__(id)

    def run(self):
        for i in range(int(1e+8)):
            pass


if __name__ == "__main__":
    tasks = [SampleTask(i+1) for i in range(10)]
    swarm = Swarm(n_bees=3)
    results = swarm.work(tasks)
    swarm.close()
    print(results)