from typing import Any, Dict

from ml_flow import Flow, GridSearch
from ml_hive import Swarm, Task

# create all tasks
dataset_fetch_task = DatasetFetchTask(1)
pre_process_task = GridSearch(task=PreProcessTask(2), gs_config=pre_process_config)  # wrap with GridSearch wrapper
featurize_task_1 = TFIDFFeaturizeTask(3)
featurize_task_2 = GridSearch(task=GloveFeaturizeTask(4), gs_config=featurize_config)  # wrap with GridSearch wrapper
train_task = GridSearch(TrainTask(5), gs_config=train_config)   # wrap with GridSearch wrapper
evaluate_task = EvaluateTask(6)


# at this moment, you are wondering what does a GridSearch wrapper do
# it still behaves like a task so that they can be attached to other tasks
# additionally, it just holds grid search config in it
# so that flow can use it when it wants to expands task

# dependencies between tasks (just like we did before)
# note wrapped task can also be attached as predecessor/successor
pre_process_task.requires([dataset_fetch_task])
featurize_task_1.requires([pre_process_task])
featurize_task_2.requires([pre_process_task])
train_task.requires([dataset_fetch_task, featurize_task_1, featurize_task_2])
evaluate_task.requires([dataset_fetch_task, featurize_task_1, featurize_task_2, train_task])

# all tasks
tasks = [dataset_fetch_task,
         pre_process_task,
         featurize_task_1, featurize_task_2,
         train_task,
         evaluate_task]

# swarm
swarm = Swarm(n_bees=10, resources=resource_list)

# create the flow with swarm
with Flow(swarm=swarm) as flow:
    # flow takes the tasks and split the tasks that are wrapped using GridSearch
    # and expands them into individual tasks
    # automatically configures dependencies based on provided task dependencies

    # after expanding, it has the list of tasks
    # which will be run internally using using the specified swarm
    flow.run(tasks)
