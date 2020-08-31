from busy_bee import Swarm, Task
from typing import List


class MyTask(Task):
    def __init__(self, id_: int, pre_task_ids: List[int], post_task_ids: List[int]):
        super().__init__(id_, pre_task_ids, post_task_ids)

    def run(self):
        return self.id_


def main():
    # tasks that I want to run
    # 1 -> 2 -> 4
    #   \- 3 /
    # 5
    tasks = [
        MyTask(1, pre_task_ids=[], post_task_ids=[2, 3]),
        MyTask(2, pre_task_ids=[1], post_task_ids=[4]),
        MyTask(3, pre_task_ids=[1], post_task_ids=[4]),
        MyTask(4, pre_task_ids=[2, 3], post_task_ids=[]),
        MyTask(5, pre_task_ids=[], post_task_ids=[]),
    ]

    with Swarm(n_bees=3, refresh_every=5) as swarm:
        results = swarm.work(tasks)
    print(results)


if __name__ == "__main__":
    main()
