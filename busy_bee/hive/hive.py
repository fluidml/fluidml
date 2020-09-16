import multiprocessing
from multiprocessing import Manager, set_start_method, Queue
import random
from types import TracebackType
from typing import Optional, Type, List, Dict

from networkx import DiGraph, shortest_path_length

from busy_bee.common.logging import Console
from busy_bee.common.task import Task, Resource
from busy_bee.hive.bee import BusyBee, QueenBee
from busy_bee.hive.honeycomb import ResultsStorage


class Swarm:
    def __init__(self,
                 n_bees: Optional[int] = None,
                 resources: Optional[List[Resource]] = None,
                 results_storage: Optional[ResultsStorage] = None,
                 start_method: str = 'spawn',
                 refresh_every: int = 1,
                 exit_on_error: bool = True):

        set_start_method(start_method, force=True)

        self.n_bees = n_bees if n_bees else multiprocessing.cpu_count()
        self.resources = Swarm._allocate_resources(self.n_bees, resources)
        self.manager = Manager()
        self.scheduled_queue = Queue()
        self.running_queue = self.manager.list()
        self.done_queue = self.manager.list()
        self.results = self.manager.dict()
        self.results_storage = results_storage
        self.exception = self.manager.dict()
        self.tasks: Dict[int, Task] = {}

        # queen bee for tracking
        self.busy_bees = [QueenBee(done_queue=self.done_queue,
                                   tasks=self.tasks,
                                   exception=self.exception,
                                   exit_on_error=exit_on_error,
                                   refresh_every=refresh_every)]
        # worker bees for task exection
        self.busy_bees.extend([BusyBee(bee_id=i,
                                       resource=self.resources[i],
                                       scheduled_queue=self.scheduled_queue,
                                       running_queue=self.running_queue,
                                       done_queue=self.done_queue,
                                       tasks=self.tasks,
                                       exception=self.exception,
                                       exit_on_error=exit_on_error,
                                       results=self.results,
                                       results_storage=self.results_storage) for i in range(self.n_bees)])

    def __enter__(self):
        return self

    def __exit__(self,
                 exc_type: Optional[Type[BaseException]],
                 exc_val: Optional[BaseException],
                 exc_tb: Optional[TracebackType]):
        self.close()

    @staticmethod
    def _allocate_resources(n_bees: int, resources: List[Resource]) -> List[Resource]:
        if not resources:
            resources = [None] * n_bees
        elif len(resources) != n_bees:
            # we assign resources to bees uniformly (from uniform distribution)
            resources = random.choices(resources, k=n_bees)
        return resources

    @staticmethod
    def _get_entry_point_tasks(tasks: List[Task]) -> Dict[int, str]:
        entry_task_ids = {}
        for task in tasks:
            if len(task.predecessors) == 0:
                entry_task_ids[task.id_] = task.name
        return entry_task_ids

    def _simplify_results(self) -> Dict[str, List[Dict]]:
        results = {}
        for task_name, run_tasks in self.results.items():
            results[task_name] = []
            for task_output in run_tasks.values():
                results[task_name].append(task_output)
        return results

    def _add_config_to_tasks(self):
        task_graph = self._create_task_graph()

        for task in self.tasks.values():
            config = self._extract_kwargs_from_ancestors(task=task, task_graph=task_graph)
            task.unique_config = config

    def _extract_kwargs_from_ancestors(self, task: Task, task_graph: DiGraph) -> Dict[str, Dict]:
        ancestor_task_ids = list(shortest_path_length(task_graph, target=task.id_).keys())[::-1]
        config = {self.tasks[id_].name: self.tasks[id_].kwargs for id_ in ancestor_task_ids}
        return config

    def _create_task_graph(self) -> DiGraph:
        task_graph = DiGraph()
        for task in self.tasks.values():
            for pred_task in task.predecessors:
                task_graph.add_edge(pred_task.id_, task.id_)
        return task_graph

    def work(self, tasks: List[Task]):
        # get entry point task ids
        entry_point_tasks: Dict[int, str] = self._get_entry_point_tasks(tasks)

        # also update the current tasks
        for task in tasks:
            self.tasks[task.id_] = task

        # add unique task configs to tasks
        self._add_config_to_tasks()

        # add entry point tasks to the job queue
        for task_id, task_name in entry_point_tasks.items():
            Console.get_instance().log(f'Swarm scheduling task {task_name}-{task_id}.')
            self.scheduled_queue.put(task_id)

        # start the workers
        for bee in self.busy_bees:
            bee.start()

        # wait for them to finish
        for bee in self.busy_bees:
            bee.join()

        # if an exception was raised by a child process, re-raise it again in the parent.
        if self.exception:
            raise self.exception['message']

        results: Dict[str, List[Dict]] = self._simplify_results()
        return results

    def close(self):
        self.results = self.manager.dict()
        self.tasks = self.manager.dict()
        self.done_queue = self.manager.dict()
        for bee in self.busy_bees:
            bee.close()
