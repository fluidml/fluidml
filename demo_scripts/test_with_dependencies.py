from busy_bee import Swarm, Task
from typing import Dict, Any


class MyTask(Task):
    def __init__(self, id_: int, name: str):
        super().__init__(id_, name)

    def run(self, results: Dict[str, Any]) -> Dict[str, Any]:
        print(f"Results from parents of task {self.id_}: {results}")
        return {"result": self.id_}


def main():
    # tasks that I want to run
    # 1 -> 2 -> 4
    #   \- 3 /
    # 5

    # create tasks
    task_1 = MyTask(1, "MyTask")
    task_2 = MyTask(2, "MyTask")
    task_3 = MyTask(3, "MyTask")
    task_4 = MyTask(4, "MyTask")
    task_5 = MyTask(5, "MyTask")

    # add dependencies
    task_2.requires([task_1])
    task_3.requires([task_1])
    task_4.requires([task_2, task_3])

    # final list of tasks
    tasks = [task_1, task_2, task_3, task_4, task_5]

    with Swarm(n_bees=3, refresh_every=5) as swarm:
        results = swarm.work(tasks)
    print(results)


if __name__ == "__main__":
    main()
