<div align="center">
<img src="logo/fluid_ml_logo.png" width="400px">

*Develop ML models fluently with no boilerplate code. Focus only on your models and not the boilerplate!*
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

## Getting Started

### Installation
1. Clone the repository,
2. Navigate into the cloned directory (contains the setup.py file),
3. Execute ``` $ pip install . ```

### Minimal Example
This minimal toy example showcases how to get started with FluidML.
For real machine learning examples check the "Examples" section.

1. First we import fluidml
```Python
from fluidml.common import Task, Resource
from fluidml.flow import Flow
from fluidml.flow import GridTaskSpec, TaskSpec
from fluidml.swarm import Swarm
from fluidml.storage import LocalFileStore
```

2. Next we define some toy machine learning tasks. A Task can be implemented as function or as class. 
Here we go with task functions. Each tasks first two arguments are
    * results: Stores task inputs generated from predecessor tasks,
    * resources: Stores global resources like seeds, etc.
    
The remaining arguments are external task configuration parameters.
```Python
def parse(results: Dict, resource: Resource, in_dir: str):
    return {}


def preprocess(results: Dict, resource: Resource, pipeline: List[str], abc: List[int]):
    return {}


def featurize_glove(results: Dict, resource: Resource, type_: str, batch_size: int):
    return {}


def featurize_tfidf(results: Dict, resource: Resource, type_: str, batch_size: int):
    return {}


def train(results: Dict, resource: Resource, model, dataloader, evaluator, optimizer, num_epochs):
    return {}


def evaluate(results: Dict, resource: Resource, metric: str):
    return {}
```



## Examples


## Citation

