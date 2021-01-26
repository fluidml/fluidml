<div align="center">
<img src="logo/fluid_ml_logo.png" width="400px">

_Develop ML pipelines fluently with no boilerplate code. Focus only on your tasks and not the boilerplate!_

---

[![Python 3.7](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CircleCI](https://circleci.com/gh/fluidml/fluidml/tree/main.svg?style=shield)](https://circleci.com/gh/fluidml/fluidml/tree/main)
[![codecov](https://codecov.io/gh/fluidml/fluidml/branch/main/graph/badge.svg?token=XG4UDWF8RE)](https://codecov.io/gh/fluidml/fluidml)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg)](https://github.com/fluidml/fluidml/blob/main/CODE_OF_CONDUCT.md)
</div>

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

---

## Getting Started

### Installation

1. Clone the repository,
2. Navigate into the cloned directory (contains the setup.py file),
3. Execute `$ pip install .`

**Note:** To run demo examples, execute `$ pip install .["examples"]` to install the additional requirements.

### Minimal Example

This minimal toy example showcases how to get started with FluidML.
For real machine learning examples, check the "Examples" section below.

1. **Basic imports:** First we import necessary classes from FluidML

```Python
from fluidml import Flow, Swarm
from fluidml.common import Task, Resource
from fluidml.flow import GridTaskSpec, TaskSpec
from fluidml.storage import MongoDBStore, LocalFileStore, ResultsStore
```

2. **Define Tasks:** Next, we define some toy machine learning tasks. A Task can be implemented as a function or as a class inheriting from Task class.

In case of the class approach, each task should implement `run()` method which takes some inputs and performs the desired functionality. These inputs are actually the results from predecessor tasks and are automatically forwarded by FluidML based on registered task dependencies. If the task has any hyperparameters, they can be defiend as arguments in the constructor. Additionally, within each task, users have access to methods and attributes like `self.save()` and `self.resource` to save its result and access task resources (more on that later).

```Python
class MyTask(Task):
    def __init__(self, kwarg_1, kwarg_2):
        ...
    def run(self, result_1, result2):
        ...
```

or

```Python
def my_task(self, result_1, result_2, kwarg_1, kwarg_2, task: Task):
    ...
```

In the case of defining the task as callable, an extra task object is provided to the task,
which makes important internal attributes and functions like `task.save()` and `task.resource` available to the user.

Below, we define standard machine learning tasks such as dataset preparation, pre-processing, featurization and model training using Task classes.
Notice that:

- Each task is implemented individually and its clear what the inputs are (check arguments of `run()` method)
- Each task saves its result using `self.save(...)` by providing the object to be saved and an unique name for it. This unique name corresponds to input names in successor task definitions.

```Python
class DatasetFetchTask(Task):
    def run(self):
        ...
        self.save(obj=data_fetch_result, name='data_fetch_result')                # For InMemoryStore (default) no type_ is required
        self.save(obj=data_fetch_result, name='data_fetch_result', type_='json')  # type_ is required for LocalFileStore, MongoDBStore


class PreProcessTask(Task):
    def __init__(self, pre_processing_steps: List[str]):
        super().__init__()
        self._pre_processing_steps = pre_processing_steps

    def run(self, data_fetch_result):
        ...
        self.save(obj=pre_process_result, name='pre_process_result')


class TFIDFFeaturizeTask(Task):
    def __init__(self, min_df: int, max_features: int):
        super().__init__()
        self._min_df = min_df
        self._max_features = max_features

    def run(self, pre_process_result):
        ...
        self.save(obj=tfidf_featurize_result, name='tfidf_featurize_result')


class GloveFeaturizeTask(Task):
    def run(self, pre_process_result):
        ...
        self.save(obj=glove_featurize_result, name='glove_featurize_result')


class TrainTask(Task):
    def __init__(self, max_iter: int, balanced: str):
        super().__init__()
        self._max_iter = max_iter
        self._class_weight = "balanced" if balanced else None

    def run(self, tfidf_featurize_result, glove_featurize_result):
        ...
        self.save(obj=train_result, name='train_result')


class EvaluateTask(Task):
    def run(self, train_result):
        ...
        self.save(obj=evaluate_result, name='evaluate_result')
```

3. **Task Specifications:** Next, we can create the defined tasks with their specifications. We now only write their specifications, later these are used to create real instances of tasks by FluidML.
   For each Task specification, we also add a list of result names that the corresponding task _publishes_ and _expects_. Each published result object will be considered when results are automatically collected for a successor task.

```Python
dataset_fetch_task = TaskSpec(task=DatasetFetchTask, publishes=['data_fetch_result'])
pre_process_task = TaskSpec(task=PreProcessTask,
                            task_kwargs={
                                "pre_processing_steps": ["lower_case", "remove_punct"]},
                            expects=['data_fetch_result'],
                            publishes=['pre_process_result'])
featurize_task_1 = TaskSpec(task=GloveFeaturizeTask,
                            expects=['pre_process_result'],
                            publishes=['glove_featurize_result'])
featurize_task_2 = TaskSpec(task=TFIDFFeaturizeTask, task_kwargs={"min_df": 5, "max_features": 1000},
                            expects=['pre_process_result'],
                            publishes=['tfidf_featurize_result'])
train_task = TaskSpec(task=TrainTask, task_kwargs={"max_iter": 50, "balanced": True},
                      expects=['glove_featurize_result', 'tfidf_featurize_result'],
                      publishes=['train_result'])
evaluate_task = TaskSpec(task=EvaluateTask, expects=['train_result'], publishes=['evaluate_result'])
```

4. **Registering task dependencies:** Here we create the task graph by connecting the tasks together by specifying predecessors for a task. For each task spec, you can specify a list of predecessors using `requires()` method.

```Python
pre_process_task.requires([dataset_fetch_task])
featurize_task_1.requires([pre_process_task])
featurize_task_2.requires([pre_process_task])
train_task.requires([dataset_fetch_task, featurize_task_1, featurize_task_2])
evaluate_task.requires([dataset_fetch_task, featurize_task_1, featurize_task_2, train_task])
```

5. **Run tasks using Flow:** Now that we have all the tasks specified, we can just run the task graph. For that we have to create an instance of Swarm class, by specifying number of workers (`n_dolphins` :wink: ). Additionally, you can pass a list of resources which are made available to the tasks (eg. GPU IDs) after balancing them.

Next, you can create an instance of the flow class and run the tasks utilizing one of our persistent ResultsStores, which defaults to InMemoryStore if no store is provided to Flow (see below for details). Flow under the hood, constructs a task graph and executes them using provided resources in swarm.

```Python
tasks = [dataset_fetch_task, pre_process_task, featurize_task_1,
         featurize_task_2, train_task, evaluate_task]

with Swarm(n_dolphins=2,
           return_results=True,
           verbose=True) as swarm:
    flow = Flow(swarm=swarm,
                results_store=None)
    results = flow.run(tasks)
```

6. **Task Results:** Results of all the tasks are returned by `flow.run()`. Users can access it via task names, for e.g. `results["EvaluationTask"]`.

---

### Results Store/Caching

By default, results of tasks are stored in an `InMemoryStore`, which might be impractical for large datasets/models. Also, the results are not persistent. To have persistent storage, FluidML provides two fully implemented `ResultsStore` namely `LocalFileStore` and `MongoDBStore`.

Additionally, users can provide their own results store to `Swarm` by inheriting from `ResultsStore` class and implementing `load()` and `save()`. Note, these methods rely on task name and its config parameters, which act as lookup-key for results. In this way, tasks are skipped by FluidML when task results are already available for the given config. But users can override and force execute tasks by passing `force` parameter to the `Flow`.

```Python
class MyResultsStore(ResultsStore):
    def load(self, name: str, task_name: str, task_unique_config: Dict) -> Optional[Any]:
        """ Query method to load an object based on its name, task_name and task_config if it exists """
        raise NotImplementedError

    @abstractmethod
    def save(self, obj: Any, name: str, type_: str, task_name: str, task_unique_config: Dict, **kwargs):
        """ Method to save/update any artifact """
        raise NotImplementedError
```

### Grid Search

Users can easily enable grid search for their tasks with just one line of code. To enable grid search on a particular task, we just have to wrap it with `GridTaskSpec` instead of `TaskSpec`.

```Python
train_task = GridTaskSpec(task=TrainTask, 
                          gs_config={"max_iter": [50, 100], 
                                     "balanced": [True, False], 
                                     "layers": [[50, 100, 50]]})
```

That's it! Internally, Flow would expand this task into 4 tasks with provided combinations of `max_iter` and `balanced`. Internally all values of type `List` will be unpacked to form grid search combinations. If a list itself is an argument and should not be unpacked, it has to be wrapped again in a list. That is why `layers` is not considered for different grid search realizations. Further, any successor tasks (for instance, evaluate task) in the task graph will also be automatically expanded. Therefore, in our example, we would have 4 evaluate tasks, each one corresponding to the 4 train tasks.

## Examples

For real machine learning pipelines including grid search implemented with FluidML, check our
Jupyter Notebook tutorials:

- [Transformer based Sequence to Sequence Translation (PyTorch)](https://github.com/fluidml/fluidml/blob/main/examples/pytorch_transformer_seq2seq_translation/transformer_seq2seq_translation.ipynb)
- [Multi-class Text Classification (Sklearn)](https://github.com/fluidml/fluidml/blob/main/examples/sklearn_text_classification/sklearn_text_classification.ipynb)

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
