from collections import defaultdict
from typing import List, Any, Dict

from busy_bee.hive import Swarm, Task
from busy_bee.flow.gs_task import GridSearch
from busy_bee.flow.create_run_configs import create_run_configs


class Flow:
    def __init__(self,
                 swarm: Swarm,
                 tasks: List[Task]):
        self._swarm = swarm
        self._tasks = tasks

    def get_list_of_tasks(self):

        # we just have to convert tasks/task_names to graphs
        # and get the task expansion order
        tasks_to_expand = []

        # expand the tasks in the topological order
        expanded_tasks_by_name = defaultdict(list)
        task_id = 0
        # for each task to expand
        for task in tasks_to_expand:
            # for each predecessor task
            for predecessor in task.predecessors:
                # for task in expanded tasks of my predecessor
                for task in id_to_task_dict[predecessor.name]:
                    task_configs = task.task_configs if isinstance(task, GridTaskSpec) else [task.kwargs]
                    # expand the current task config
                    for kwargs in task_configs:
                        task = task.wrapped_task(id_: task_id, name, kwargs)
                        task.requires(predecessor)
                        expanded_tasks_by_name[task.name].append(task)
                        task_id += 1

        # I think at this point, we would have our final list of tasks in expanded_tasks_by_name


                # for kwargs in task.task_configs:
                #     task = {'name': task.name,
                #             'task': task.wrapped_task,
                #             'kwargs': kwargs}

                #     dep_kwargs = []
                #     for predecessor in task.predecessors:
                #         if isinstance(predecessor, GridSearch):
                #             for dep_kwarg in predecessor.task_configs:
                #                 dep_kwargs.append(config[dep])
                #     task['dep_kwargs'] = dep_kwargs


    def run(self) -> Dict[str, Dict[str, Any]]:
        """
        Runs the provided tasks

        Args:
            tasks (List[Task]): tasks to run

        Returns:
            Dict[str, Dict[str, any]] - a nested dictionary of results
        """

        for task in tasks:
            if isinstance(task, GridSearch):
                expanded_tasks: List[Task] = task.expand()

                unique_task_configs = create_run_configs(task.gs_config)

        # split grid search config in individual run configs
        single_run_configs = create_run_configs(config_grid_search=config)

        # get all unique tasks, taking task dependency and kwargs dependency into account
        id_to_task_dict = {}
        task_id = 0
        for config in single_run_configs:
            for task_name, dependencies in pipeline.items():
                # create task object, holding its kwargs (config parameters)
                task = {'name': task_name,
                        'task': TASK_TO_CALLABLE[task_name],
                        'kwargs': config[task_name]}

                # add kwargs of dependent tasks for later comparison
                dep_kwargs = []
                for dep in dependencies:
                    dep_kwargs.append(config[dep])
                task['dep_kwargs'] = dep_kwargs

                # check if task object is already included in task dictionary, skip if yes
                if task in id_to_task_dict.values():
                    continue

                # if no, add task to task dictionary
                id_to_task_dict[task_id] = task

                # increment task_id counter
                task_id += 1

        # create graph object, to model task dependencies
        graph = nx.DiGraph()
        for id_i, task_i in id_to_task_dict.items():
            # for task_i, get kwargs of dependent tasks
            dep_kwargs = task_i['dep_kwargs']
            for id_j, task_j in id_to_task_dict.items():
                # for task_j, get kwargs
                kwargs = task_j['kwargs']
                # create an edge between task_j and task_i if task_j depends on task_i
                if kwargs in dep_kwargs:
                    graph.add_edge(id_j, id_i)

        # convert task_dict to an abstract Task class -> Interface for swarm
        id_to_task = {}
        for id_, task_dict in id_to_task_dict.items():
            task = MyTask(id_=id_,
                          name=task_dict['name'],
                          task=task_dict['task'],
                          kwargs=task_dict['kwargs'])
            id_to_task[id_] = task

        # Register dependencies
        tasks = []
        for id_, task in id_to_task.items():
            pre_task_ids = list(graph.predecessors(id_))
            task.requires([id_to_task[id_] for id_ in pre_task_ids])
            tasks.append(task)

        # run swarm
        with Swarm(n_bees=3, refresh_every=5, exit_on_error=True) as swarm:
            results = swarm.work(tasks=tasks)

        # 1. first expand the tasks that are grid searcheable

        # 2. also, take care of their dependencies

        # 3. get a final list of tasks

        # 4. run the tasks through swarm

        # 5. return results
