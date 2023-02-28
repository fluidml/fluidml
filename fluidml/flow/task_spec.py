import inspect
import json
from typing import Dict, Any, Optional, List, Union, Callable, Type

from fluidml.common import DependencyMixin, Task, TaskData
from fluidml.common.utils import remove_prefixed_keys_from_dict, remove_prefix_from_dict, remove_none_from_dict
from fluidml.flow.config_expansion import expand_config


class TaskSpec(TaskData, DependencyMixin):
    def __init__(
        self,
        task: Union[Type["Task"], Callable],
        config: Optional[Dict[str, Any]] = None,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        reduce: Optional[bool] = None,
        expand: Optional[str] = None,
        config_ignore_prefix: Optional[str] = None,
        config_group_prefix: Optional[str] = None,
    ):
        """A class to hold specification of a plain task.

        Args:
            task: Task class
            config: Task configuration parameters that are used while instantiating. Defaults to ``None``.
            additional_kwargs: Additional kwargs provided to the task.
            name: Name of the task. Defaults to None.
            reduce: A boolean indicating whether this is a reduce task. Defaults to None.
            expand: Config expansion method, choose between "zip" and "product".
            config_ignore_prefix: A config key prefix, e.g. "_". Prefixed keys will be not included in the
                "unique_config", which is used to determine whether a run has been executed or not.
            config_group_prefix: A config grouping prefix, to indicate that to parameters are grouped and expanded
                using the "zip" method. The grouping prefix enables the "zip" expansion of specific parameters, while
                all remaining grid parameters are expanded via "product".
                Example:
                    cfg = {"a": [1, 2, "@x"], "b": [1, 2, 3], "c": [1, 2, "@x"]

                    Without grouping "product" expansion would yield: 2 * 2 * 3 = 12 configs.
                    With grouping "product" expansion yields : 2 * 3 = 6 configs, since the grouped parameters are
                    "zip" expanded.
        """
        DependencyMixin.__init__(self)
        TaskData.__init__(self)

        # task has to be a class object which inherits Task or it has to be a function
        if not ((inspect.isclass(task) and issubclass(task, Task)) or inspect.isfunction(task)):
            raise TypeError(
                f'{task} needs to be a Class object which inherits Task (type="type") or a function.'
                f'But it is of type "{type(task)}".'
            )

        # "reduce" can only be set to "True" if "expand" is "None".
        if reduce and expand:
            raise ValueError(f'"reduce" can only be set to "True" if "expand" is "None".')

        # we assure that the provided config is json serializable since we use json to later store the config
        config = json.loads(json.dumps(config)) if config is not None else {}

        # set name and additional kwargs
        name = name if name is not None else task.__name__

        additional_kwargs = additional_kwargs if additional_kwargs is not None else {}

        # TaskSpec attributes
        self.task = task
        self.expand_fn = expand
        self.config_group_prefix = config_group_prefix
        self.config_ignore_prefix = config_ignore_prefix

        # TaskData attributes
        self.name = name
        self.unique_name = name  # gets overwritten in case multiple instances of this task exists in the graph
        self.config: Dict = config
        self.additional_kwargs: Dict = additional_kwargs
        self.reduce = reduce
        # dynamically retrieve expected arguments from task implementation
        self.expects = get_expected_args_from_run_signature(task, config, additional_kwargs)

    def expand(self) -> List["TaskSpec"]:
        return [
            TaskSpec(
                task=self.task,
                config=config,
                additional_kwargs=self.additional_kwargs,
                name=self.name,
                reduce=self.reduce,
                config_group_prefix=self.config_group_prefix,
                config_ignore_prefix=self.config_ignore_prefix,
            )
            for config in expand_config(self.config, self.expand_fn, group_prefix=self.config_group_prefix)
        ]

    def prepare_config(self):
        # remove keys with None values as well as prefixed keys to ignore from config
        # creates a new relevant_config object
        relevant_config = remove_prefixed_keys_from_dict(
            remove_none_from_dict(self.config), prefix=self.config_ignore_prefix
        )

        # mutate self.config object by removing the ignore prefix from all keys
        self.config = remove_prefix_from_dict(self.config, prefix=self.config_ignore_prefix)

        return relevant_config


def get_expected_args_from_run_signature(
    task: Union[Type["Task"], Callable], config: Dict, additional_kwargs: Dict
) -> Dict[str, inspect.Parameter]:
    if inspect.isclass(task):
        task_all_arguments = dict(inspect.signature(task.run).parameters)
        expected_inputs = {
            arg: value
            for arg, value in task_all_arguments.items()
            if value.kind.name not in ["VAR_POSITIONAL", "VAR_KEYWORD"] and value.name != "self"
        }
    elif inspect.isfunction(task):
        task_all_arguments = dict(inspect.signature(task).parameters)
        task_extra_arguments = list(config) + list(additional_kwargs) + ["task"]
        expected_inputs = {
            arg: value
            for arg, value in task_all_arguments.items()
            if arg not in task_extra_arguments and value.kind.name not in ["VAR_POSITIONAL", "VAR_KEYWORD"]
        }
    else:
        # cannot be reached, check has been made in TaskSpec.
        raise TypeError

    return expected_inputs
