from abc import ABC, abstractmethod
from multiprocessing import Manager
from typing import List, Dict, Any, Callable, Optional
import json
import os
import pickle


# class Artifact:
#     name: str                                       # a unique name for the result (eg. raw_dataset)
#     data: Any                                       # any data which is to be stored
#     save_fn: Callable[[str, Dict, Artifact], None]  # a callable that is specified by user how to save the data
#     load_fn: Callable[[str, Dict, str], Dict]       # a callable that is specified by user how to load the data back
#     save_kwargs: Optional[Dict]
#     load_kwargs: Optional[Dict]


class ResultsStore(ABC):

    @abstractmethod
    def get_results(self, task_name: str, unique_config: Dict, publishes: List[str]) -> Optional[Dict]:
        """ Query method to get the results if they exist already """
        raise NotImplementedError


class LocalFileStore(ResultsStore):
    def get_results(self, task_name: str, unique_config: Dict, publishes: List[str]) -> Optional[Dict]:
        # get run dir with task_name and unique_config
        run_dir = self._get_run_dir(task_name=task_name, unique_config=unique_config)

        results = {}
        for item_name in publishes:
            # load meta data
            try:
                meta_data = pickle.load(open(run_dir + f"{item_name}_meta.p", "wb"))
            except FileNotFoundError:
                raise FileNotFoundError(f'{item_name} not saved.')

            load_fn = meta_data['load_fn']
            name = meta_data['name']

            # load the saved object
            obj = load_fn(run_dir, name)

            # store obj in results
            results[item_name] = obj

        return results

    def save_json(self, name: str, d: Dict, task_name: str, unique_config: Dict):
        # get run dir with task_name and unique_config
        run_dir = self._get_run_dir(task_name=task_name, unique_config=unique_config)

        # save dict
        json.dump(d, open(os.path.join(run_dir, f'{name}.json'), "w"))

        # save load info
        meta_data = {"name": name, "load_fn": self._load_json}
        pickle.dump(meta_data, open(run_dir + f"{name}_meta.p", "wb"))

    def _load_json(self, run_dir: str, name: str) -> Dict:
        return json.load(open(os.path.join(run_dir, f'{name}.json'), "w"))

    def save_torch(self, name: str, model: nn.Module, task_name: str, unique_config: Dict):
        # get run dir with task_name and unique_config
        run_dir = self._get_run_dir(task_name=task_name, unique_config=unique_config)

        # save model
        torch.save(model.state_dict(), os.path.join(run_dir, f'{name}.pt'))

        # save load info
        meta_data = {"name": name, "load_fn": self._load_torch}
        pickle.dump(meta_data, open(run_dir + f"{name}_meta.p", "wb"))

    def _load_torch(self, run_dir: str, name: str):
        return torch.load(os.path.join(run_dir, f'{name}.pt'))


class MongoDBStore(ResultsStore):
    pass


class InMemoryStore(ResultsStore):
    def __init__(self):
        self._memory_store = Manager().dict()

    def get_results(self, task_name: str, unique_config: Dict, publishes: List[str]) -> Optional[Dict]:
        # if no store is used, default to get results from in-memory
        if task_name not in self._memory_store:
            return None

        for task_sweep in self._memory_store[task_name]:
            if task_sweep["config"] == unique_config:
                results = {}
                for item_name in publishes:
                    try:
                        results[item_name] = task_sweep['results'][item_name]
                    except KeyError:
                        raise KeyError(f'{item_name} not saved.')

                return results

    def save(self, name: str, obj: Any, task_name: str, unique_config: Dict):
        """ In-memory save function.
        Adds individual object to in-memory store (multiprocessing manager dict).
        """
        if task_name not in self._memory_store:
            self._memory_store[task_name] = []

        existing_task_results = self._memory_store[task_name]
        sweep_exists = False
        for task_sweep in existing_task_results:
            if task_sweep['config'] == unique_config:
                task_sweep['results'][name] = obj
                sweep_exists = True
                break

        if not sweep_exists:
            new_task_sweep = {'results': {name: obj},
                              'config': unique_config}
            existing_task_results.append(new_task_sweep)

        self._memory_store[task_name] = existing_task_results


class Train(Task):

    publishes = ["torch_model", "dataset"]

    # can be an instance of any user defined store (LocalFileStore, MongoDBStore, etc.)
    # Defaults to InMemoryStore
    # As before the user provides his own results store inheriting from ResultStorage to Swarm.
    self.store = store

    def run(self):
        # string_result = Artifact("info", {"some_info": 'bla'}, self.store.save_json, self.store.load_json)
        # self.results_store.save(string_result)

        # No need for ResultItem class in this way. More direct way for user to save arbitrary objects
        some_info_dict = {'a': 1, 'b': 2}
        self.store.save_json(name='info', obj=some_info_dict, task_name=task.name, unique_config=task.config)

        some_model = nn.GRU()
        self.store.save_torch(name='model', obj=some_model, task_name=task.name, unique_config=task.config)

        # if user always wants to call the same save fn regardless of type, he can provide type as argument
        # self.store.save(name='info', obj=some_info_dict, task_name=task.name, unique_config=task.config, type='json')
        # and inside of his store class he registers different functions to type keys
        # e.g. 'json': self.save_json
        #      'torch': self.save_torch
        # and the general self.store.save() fn queries and calls the corresponding fn by type (e.g. 'json' -> save_json)


class Evaluate(Task):
    def __init__(self):
        super().__init__()

    def run(self, torch_model, dataset):
        pass


train = TaskSpec(train_fn, publishes=["torch_model", "dataset"])