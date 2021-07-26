# BEFORE: pipelining intertwined with business logic
The model/pipeline takes several components as their inputs
and their dependencies are statically defined in their code

```python
model = MyModel(pre_processors: List[Any] = ..,
                classifier: torch.nn.Module = ..,
                post_processors: List[Any] = ..)
model_outputs = model.run(model_inputs)
```
##  Problems with this setup
1) The dependencies are statically defined, so any variation would result in coding a new model altogether
2) Components are designed in such a way that they work with only within a particular workflow or pipeline ie. they do not have consistent interfaces. So, creating a new pipeline would also require refactoring of the components limiting reusability.
3) User has to implement serialization functions to load and save the pipeline and its components 


# AFTER: With FluidML's pipeline

Pipeline in FluidML provides a unified API to create components and build pipelines by wiring them together.


```python
from fluidml import Component


class PreProcessor(Component):
    def run(self, inputs: List[text]) -> torch.Tensor:
        ...

class PostProcessor(Component):
    def run(self, inputs: torch.Tensor) -> Dict:
        ...

class Classifier(Component):
    def run(self, inputs: torch.Tensor) -> torch.Tensor:
        ...


# instantiate components
pre_processor = PreProcessor(..)
post_processor = PostProcessor(..)
classifier = Classifier(..)

# dependencies
classifier.requires([pre_processor])
post_processor.requires([classifier])

# final pipeline
pipeline = Pipeline([pre_processor, classifier, post_processor])

# run pipeline 
pipeline.run(mode_inputs)

# save pipeline
pipeline.save(file_like_object)

# load pipeline (also in an env where component code is not present)
pipeline = Pipeline.load(file_like_object)

```
## Basic functionalities for initial prototype:
1) APIs for Component and Pipeline
2) Unify training and inference pipelines (by having different modes for inputs and components)
3) Basic validation of component dependencies based on type-checking of inputs and outputs
4) Serialization of pipelines
