from multiprocessing import Process, Queue


class BusyBee(Process):
    def __init__(self, bee_id: int, task_queue: Queue):
        super().__init__(target=self.work, args=(task_queue,))
        self.bee_id = bee_id

    def work(self, task_queue: Queue):
        while True:
            # get the next task in the queue
            task = task_queue.get()

            if task == -1:
                print(f"Bee {self.bee_id} leaving the swarm!")
                return

            print(f'Bee {self.bee_id} started running task {task.id}')

            # run the task
            task.run()

            print(f'Bee {self.bee_id} completed running task {task.id}')
