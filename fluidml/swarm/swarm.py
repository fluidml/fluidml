# from collections import defaultdict
import multiprocessing
from multiprocessing import Manager, set_start_method, Queue, Lock
import random
from types import TracebackType
from typing import Optional, Type, List, Dict  # , Tuple, Union

# from networkx import DiGraph, shortest_path_length

from fluidml.common.logging import Console
from fluidml.common.task import Task, Resource
from fluidml.swarm.dolphin import Dolphin, Orca
from fluidml.swarm.storage import ResultsStorage


class Swarm:
    def __init__(self,
                 n_dolphins: Optional[int] = None,
                 resources: Optional[List[Resource]] = None,
                 results_storage: Optional[ResultsStorage] = None,
                 start_method: str = 'spawn',
                 refresh_every: int = 1,
                 exit_on_error: bool = True):

        set_start_method(start_method, force=True)

        self.n_dolphins = n_dolphins if n_dolphins else multiprocessing.cpu_count()
        self.resources = Swarm._allocate_resources(self.n_dolphins, resources)
        self.manager = Manager()
        self.scheduled_queue = Queue()
        self.lock = Lock()
        self.running_queue = self.manager.list()
        self.done_queue = self.manager.list()
        self.results = self.manager.dict()
        self.results_storage = results_storage
        self.exception = self.manager.dict()
        self.tasks: Dict[int, Task] = self.manager.dict()

        # orca worker for tracking
        self.dolphins = [Orca(done_queue=self.done_queue,
                              tasks=self.tasks,
                              exception=self.exception,
                              exit_on_error=exit_on_error,
                              refresh_every=refresh_every)]
        # dolphin workers for task exection
        self.dolphins.extend([Dolphin(id_=i,
                                      resource=self.resources[i],
                                      scheduled_queue=self.scheduled_queue,
                                      running_queue=self.running_queue,
                                      done_queue=self.done_queue,
                                      lock=self.lock,
                                      tasks=self.tasks,
                                      exception=self.exception,
                                      exit_on_error=exit_on_error,
                                      results=self.results,
                                      results_storage=self.results_storage) for i in range(self.n_dolphins)])

    def __enter__(self):
        return self

    def __exit__(self,
                 exc_type: Optional[Type[BaseException]],
                 exc_val: Optional[BaseException],
                 exc_tb: Optional[TracebackType]):
        self.close()

    @staticmethod
    def _allocate_resources(n_dolphins: int, resources: List[Resource]) -> List[Resource]:
        if not resources:
            resources = [None] * n_dolphins
        elif len(resources) != n_dolphins:
            # we assign resources to bees uniformly (from uniform distribution)
            resources = random.choices(resources, k=n_dolphins)
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

    # @staticmethod
    # def _update_merge(d1: Dict, d2: Dict) -> Union[Dict, Tuple]:
    #     if isinstance(d1, dict) and isinstance(d2, dict):
    #         # Unwrap d1 and d2 in new dictionary to keep non-shared keys with **d1, **d2
    #         # Next unwrap a dict that treats shared keys
    #         # If two keys have an equal value, we take that value as new value
    #         # If the values are not equal, we recursively merge them
    #         return {**d1, **d2,
    #                 **{k: d1[k] if d1[k] == d2[k] else Swarm._update_merge(d1[k], d2[k])
    #                    for k in {*d1} & {*d2}}
    #                 }
    #     else:
    #         # This case happens when values are merged
    #         # It bundle values in a tuple, assuming the original dicts
    #         # don't have tuples as values
    #         if isinstance(d1, tuple) and not isinstance(d2, tuple):
    #             combined = d1 + (d2,)
    #         elif isinstance(d2, tuple) and not isinstance(d1, tuple):
    #             combined = d2 + (d1,)
    #         elif isinstance(d1, tuple) and isinstance(d2, tuple):
    #             combined = d1 + d2
    #         else:
    #             combined = (d1, d2)
    #         return tuple(sorted(element for i, element in enumerate(combined) if element not in combined[:i]))
    #
    # @staticmethod
    # def _reformat_config(d: Dict) -> Dict:
    #     for key, value in d.items():
    #         # if isinstance(value, list):
    #         #     d[key] = [value]
    #         if isinstance(value, tuple):
    #             d[key] = list(value)
    #         elif isinstance(value, dict):
    #             Swarm._reformat_config(value)
    #         else:
    #             continue
    #     return d
    #
    # @staticmethod
    # def _merge_configs(configs: List[Dict]) -> Dict:
    #     merged_config = configs.pop(0)
    #     for config in configs:
    #         merged_config: Dict = Swarm._update_merge(merged_config, config)
    #
    #     # merged_config: Dict = Swarm._reformat_config(merged_config)
    #     return merged_config
    #
    # def _extract_kwargs_from_ancestors(self, task: Task, task_graph: DiGraph) -> Dict[str, Dict]:
    #     ancestor_task_ids = list(shortest_path_length(task_graph, target=task.id_).keys())[::-1]
    #
    #     # task_configs = defaultdict(list)
    #     # for id_ in ancestor_task_ids:
    #     #     task_configs[self.tasks[id_].name].append(self.tasks[id_].kwargs)
    #     #
    #     # config = {name: Swarm._merge_configs(configs=configs) for name, configs in task_configs.items()}
    #     config = {self.tasks[id_].name: self.tasks[id_].kwargs for id_ in ancestor_task_ids}
    #     return config
    #
    # def _add_config_to_tasks(self):
    #     task_graph = self._create_task_graph()
    #
    #     for id_, task in self.tasks.items():
    #         config = self._extract_kwargs_from_ancestors(task=task, task_graph=task_graph)
    #         task.unique_config = config
    #         self.tasks[id_] = task
    #
    # def _create_task_graph(self) -> DiGraph:
    #     task_graph = DiGraph()
    #     for task in self.tasks.values():
    #         for pred_task in task.predecessors:
    #             task_graph.add_edge(pred_task.id_, task.id_)
    #     return task_graph

    def work(self, tasks: List[Task]):
        # get entry point task ids
        entry_point_tasks: Dict[int, str] = self._get_entry_point_tasks(tasks)

        # also update the current tasks
        for task in tasks:
            self.tasks[task.id_] = task

        # add unique task configs to tasks
        # self._add_config_to_tasks()

        # add entry point tasks to the job queue
        for task_id, task_name in entry_point_tasks.items():
            Console.get_instance().log(f'Swarm scheduling task {task_name}-{task_id}.')
            self.scheduled_queue.put(task_id)

        # start the workers
        for dolphin in self.dolphins:
            dolphin.start()

        # wait for them to finish
        for dolphin in self.dolphins:
            dolphin.join()

        # if an exception was raised by a child process, re-raise it again in the parent.
        if self.exception:
            raise self.exception['message']

        results: Dict[str, List[Dict]] = self._simplify_results()
        return results

    def close(self):
        self.results = self.manager.dict()
        self.tasks = self.manager.dict()
        self.done_queue = self.manager.dict()
        for dolphin in self.dolphins:
            dolphin.close()
