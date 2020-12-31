<div align="center">
<img src="logo/fluid_ml_logo.png" width="400px">

**Develop ML models fluently with no boilerplate code. Focus only on your models and not the boilerplate!**
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

## Examples

## Citation

