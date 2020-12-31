<div align="center">
<img src="logo/fluid_ml_logo.png" width="400px">

_Develop ML models fluently with no boilerplate code. Focus only on your models and not the boilerplate!_

</div>

[![CircleCI](https://circleci.com/gh/fluidml/fluidml/tree/main.svg?style=svg)](https://circleci.com/gh/fluidml/fluidml/tree/main)

---

**FluidML** is a lightweight framework for developing machine learning pipelines.

Developing machine learning models is a challenging process, with a wide range of sub-tasks: data collection, pre-processing, model development, hyper-parameter tuning and deployment. Each of these tasks is iterative in nature and requires lot of iterations to get it right with good performance.

Due to this, each task is generally developed sequentially, with artifacts from one task being fed as inputs to the subsequent tasks. For instance, raw datasets are first cleaned, pre-processed, featurized and stored as iterable datasets (on disk), which are then used for model training. However, this type of development can become messy and un-maintenable quickly for several reasons:

- pipeline code may be split across multiple scripts whose dependencies are not modeled explicitly
- each of this task contains boilerplate code to collect results from previous tasks (eg: reading from disk)
- hard to keep track of task artifacts and their different versions
- hyper-parameter tuning adds further complexity and boilerplate code

## Key Features

FluidML provides following functionalities out-of-the-box:

- **Task Graphs** - Create ML pipelines or task graph using simple APIs
- **Results Forwarding** - Results from tasks are automatically forwarded to downstream tasks based on dependencies
- **Parallel Processing** - Execute the task graph parallely with multi-processing
- **Grid Search** - Extend the task graph by enabling grid search on tasks with just one line of code
- **Result Caching** - Task results are cached in a results store (eg: Local File Store or a MongoDB Store) and made available for subsequent runs without executing the tasks again and again

## Getting Started

### **Installation**

1. Clone the repository,
2. Navigate into the cloned directory (contains the setup.py file),
3. Execute `$ pip install .`

### **Minimal Example**

This minimal toy example showcases how to get started with FluidML.
For real machine learning examples, check the "Examples" section.

1. **Basic imports:** First we import fluidml

```Python
from fluidml.common import Task, Resource
from fluidml.flow import Flow
from fluidml.flow import GridTaskSpec, TaskSpec
from fluidml.swarm import Swarm
from fluidml.storage import MongoDBStore, LocalFileStore, ResultsStore
```

2. **Define Tasks:** Next we define some toy machine learning tasks. A Task can be implemented as function or as class inheriting from Task class.

```Python
class MyTask(Task):
    def __init__(self, **kwargs: Dict):
        ..
    def run(self, results: Dict[str, Any], task_config: Dict[str, Any], resource: Resource):
        ..
```

or

```Python
def my_task(self, results: Dict[str, Any], task_config: Dict[str, Any], resource: Resource, **kwargs: Dict):
    ..
```

There are three arguments:

- results: contains task inputs generated from predecessor tasks
- task_config: contains config of the task in the graph (including its predecessors)
- resource: contains global resources like devices, seeds, etc.

For example, we can define our typical machine learning tasks (using Task classes)

```Python
class DatasetFetchTask(Task):
    def run(self, results: Dict[str, Any], task_config: Dict[str, Any], resource: Resource):
        return task_results


class PreProcessTask(Task):
    def __init__(self, pre_processing_steps: List[str]):
        super().__init__()
        self._pre_processing_steps = pre_processing_steps

    def run(self, results: Dict[str, Any], task_config: Dict[str, Any], resource: Resource):
        ..
        return task_results


class TFIDFFeaturizeTask(Task):
    def __init__(self, min_df: int, max_features: int):
        super().__init__()
        self._min_df = min_df
        self._max_features = max_features

    def run(self, results: Dict[str, Any], task_config: Dict[str, Any], resource: Resource):
        ..
        return task_results


class GloveFeaturizeTask(Task):
    def run(self, results: Dict[str, Any], task_config: Dict[str, Any], resource: Resource):
        ..
        return task_results


class TrainTask(Task):
    def __init__(self, max_iter: int, balanced: str):
        super().__init__()
        self._max_iter = max_iter
        self._class_weight = "balanced" if balanced else None

    def run(self, results: Dict[str, Any], task_config: Dict[str, Any], resource: Resource):
        ..
        return task_results


class EvaluateTask(Task):
    def run(self, results: Dict[str, Any], task_config: Dict[str, Any], resource: Resource):
        ..
        return task_results
```

3. **Task Specifications:** Next we can create the defined tasks with their specifications. We now only specify their specifications, later these are used to create real instances of tasks.

```Python
dataset_fetch_task = TaskSpec(task=DatasetFetchTask)
pre_process_task = TaskSpec(task=PreProcessTask, task_kwargs={
                                "pre_processing_steps": ["lower_case", "remove_punct"]})
featurize_task_1 = TaskSpec(task=GloveFeaturizeTask)
featurize_task_2 = TaskSpec(task=TFIDFFeaturizeTask, task_kwargs={"min_df": 5, "max_features": 1000})
train_task = TaskSpec(task=TrainTask, task_kwargs={"max_iter": 50, "balanced": True})
evaluate_task = TaskSpec(task=EvaluateTask)
```

4. **Task Graphs:** Create the task graph by connecting the tasks together by specifying predecessors for a task.

```Python
pre_process_task.requires([dataset_fetch_task])
featurize_task_1.requires([pre_process_task])
featurize_task_2.requires([pre_process_task])
train_task.requires([dataset_fetch_task, featurize_task_1, featurize_task_2])
evaluate_task.requires([dataset_fetch_task, featurize_task_1, featurize_task_2, train_task])
```

5. **Run tasks using Flow:** Now that we have all the tasks, we can just run the task graph. For that we have to create an instance of Swarm class, by specifying number of workers (n_dolphins ;) ). Additionally, you can pass a list of resources which are made available to the tasks (eg. GPU IDs) after balancing them.

Next, you can create an instance of the flow class and run the tasks. Flow under the hood, constructs a task graph and executes them using provided resources in swarm.

```Python
tasks = [dataset_fetch_task, pre_process_task, featurize_task_1,
         featurize_task_2, train_task, evaluate_task]

with Swarm(n_dolphins=2,
           refresh_every=10,
           return_results=True) as swarm:
    flow = Flow(swarm=swarm)
    results = flow.run(tasks)
```

6. **Task Results:** Results of all the tasks are returned by `flow.run()`. Users can access it via task names. For eg. `results["EvaluationTask"]`

### **Results Store**

By default, results of tasks are stored in an InMemoryStore, which might be impractical for large datasets/models. Also, the results are not persistent. To have persistent storage, FluidML provides two fully implemented `ResultsStore` namely `LocalFileStore` and `MongoDBStore`.

Additionally, users can provide their own results store to `Swarm` by inheriting from `ResultsStore` class and implementing `get_results()` and `save_results()`. Note, these methods rely on task name and its config parameters, which act as key for results. In this way, tasks are skipped by FluidML when task results are already available for the given config. But users can override and force execute tasks by passing `force` parameter to the `Flow`.

```Python
class ResultsStore(ResultsStore):
    def get_results(self, task_name: str, unique_config: Dict) -> Optional[Dict]:
        """ Query method to get the results if they exist already """
        pass

    @abstractmethod
    def save_results(self, task_name: str, unique_config: Dict, results: Dict):
        """ Method to save new results """
        pass

    @abstractmethod
    def update_results(self, task_name: str, unique_config: Dict, results: Dict):
        """ Method to overwrite and update existing results """
        pass
```

### **Grid Search**

Users can easily enable grid search for their tasks with just one line of code. To enable grid search on a particular task, we just have to wrap it with `GridTaskSpec` instead of `TaskSpec`.

```Python
train_task = GridTaskSpec(task=TrainTask, gs_config={
                              "max_iter": [50, 100], "balanced": [True, False]})
```

That's it! Internally, Flow would expand this task into 4 tasks with provided combinations of `max_iter` and `balanced`. Not only that, any successor tasks (for instance, evaluate task) in the task graph will also be automatically extended. Therefore, in our example, we would have 4 evaluate tasks each one corresponding to each one of 4 train tasks.

## Examples

For a real machine learning example with complete grid search, check our [tutorial notebook](https://github.com/fluidml/fluidml/blob/main/examples/sklearn/tutorial.ipynb)

## Citation

```
@article{fluid_ml,
  title = {FluidML - a lightweight framework for developing machine learning pipelines},
  author = {Ramamurthy, Rajkumar and Hillebrand, Lars},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/fluidml/fluidml}},
}
```
