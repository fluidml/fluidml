from abc import ABC, abstractmethod
from multiprocessing import Manager
from typing import List, Dict, Any, Callable, Optional
import json
import os
import pickle


import torch
import torch.nn as nn


# class Artifact:
#     name: str                                       # a unique name for the result (eg. raw_dataset)
#     data: Any                                       # any data which is to be stored
#     save_fn: Callable[[str, Dict, Artifact], None]  # a callable that is specified by user how to save the data
#     load_fn: Callable[[str, Dict, str], Dict]       # a callable that is specified by user how to load the data back
#     save_kwargs: Optional[Dict]
#     load_kwargs: Optional[Dict]


# class ResultsStore(ABC):
#
#     @abstractmethod
#     def get_results(self, task_name: str, unique_config: Dict, publishes: List[str]) -> Optional[Dict]:
#         """ Query method to get the results if they exist already """
#         raise NotImplementedError
#
#
# class LocalFileStore(ResultsStore):
#     def get_results(self, task_name: str, unique_config: Dict, publishes: List[str]) -> Optional[Dict]:
#         # get run dir with task_name and unique_config
#         run_dir = self._get_run_dir(task_name=task_name, unique_config=unique_config)
#
#         results = {}
#         for item_name in publishes:
#             # load meta data
#             try:
#                 meta_data = pickle.load(open(run_dir + f"{item_name}_meta.p", "wb"))
#             except FileNotFoundError:
#                 raise FileNotFoundError(f'{item_name} not saved.')
#
#             load_fn = meta_data['load_fn']
#             name = meta_data['name']
#
#             # load the saved object
#             obj = load_fn(run_dir, name)
#
#             # store obj in results
#             results[item_name] = obj
#
#         return results
#
#     def save_json(self, name: str, d: Dict, task_name: str, unique_config: Dict):
#         # get run dir with task_name and unique_config
#         run_dir = self._get_run_dir(task_name=task_name, unique_config=unique_config)
#
#         # save dict
#         json.dump(d, open(os.path.join(run_dir, f'{name}.json'), "w"))
#
#         # save load info
#         meta_data = {"name": name, "load_fn": self._load_json}
#         pickle.dump(meta_data, open(run_dir + f"{name}_meta.p", "wb"))
#
#     def _load_json(self, run_dir: str, name: str) -> Dict:
#         return json.load(open(os.path.join(run_dir, f'{name}.json'), "w"))
#
#     def save_torch(self, name: str, model: nn.Module, task_name: str, unique_config: Dict):
#         # get run dir with task_name and unique_config
#         run_dir = self._get_run_dir(task_name=task_name, unique_config=unique_config)
#
#         # save model
#         torch.save(model.state_dict(), os.path.join(run_dir, f'{name}.pt'))
#
#         # save load info
#         meta_data = {"name": name, "load_fn": self._load_torch}
#         pickle.dump(meta_data, open(run_dir + f"{name}_meta.p", "wb"))
#
#     def _load_torch(self, run_dir: str, name: str):
#         return torch.load(os.path.join(run_dir, f'{name}.pt'))
#
#
# class MongoDBStore(ResultsStore):
#     pass
#
#
# class InMemoryStore(ResultsStore):
#     def __init__(self):
#         self._memory_store = Manager().dict()
#
#     def get_results(self, task_name: str, unique_config: Dict, publishes: List[str]) -> Optional[Dict]:
#         # if no store is used, default to get results from in-memory
#         if task_name not in self._memory_store:
#             return None
#
#         for task_sweep in self._memory_store[task_name]:
#             if task_sweep["config"] == unique_config:
#                 results = {}
#                 for item_name in publishes:
#                     try:
#                         results[item_name] = task_sweep['results'][item_name]
#                     except KeyError:
#                         raise KeyError(f'{item_name} not saved.')
#
#                 return results
#
#     def save(self, name: str, obj: Any, task_name: str, unique_config: Dict):
#         """ In-memory save function.
#         Adds individual object to in-memory store (multiprocessing manager dict).
#         """
#         if task_name not in self._memory_store:
#             self._memory_store[task_name] = []
#
#         existing_task_results = self._memory_store[task_name]
#         sweep_exists = False
#         for task_sweep in existing_task_results:
#             if task_sweep['config'] == unique_config:
#                 task_sweep['results'][name] = obj
#                 sweep_exists = True
#                 break
#
#         if not sweep_exists:
#             new_task_sweep = {'results': {name: obj},
#                               'config': unique_config}
#             existing_task_results.append(new_task_sweep)
#
#         self._memory_store[task_name] = existing_task_results
#
#
# class Train(Task):
#
#     publishes = ["torch_model", "dataset"]
#
#     # can be an instance of any user defined store (LocalFileStore, MongoDBStore, etc.)
#     # Defaults to InMemoryStore
#     # As before the user provides his own results store inheriting from ResultStorage to Swarm.
#     self.store = store
#
#     def run(self):
#         # string_result = Artifact("info", {"some_info": 'bla'}, self.store.save_json, self.store.load_json)
#         # self.results_store.save(string_result)
#
#         # No need for ResultItem class in this way. More direct way for user to save arbitrary objects
#         some_info_dict = {'a': 1, 'b': 2}
#         self.store.save_json(name='info', obj=some_info_dict, task_name=task.name, unique_config=task.config)
#
#         some_model = nn.GRU()
#         self.store.save_torch(name='model', obj=some_model, task_name=task.name, unique_config=task.config)
#
#         # if user always wants to call the same save fn regardless of type, he can provide type as argument
#         # self.store.save(name='info', obj=some_info_dict, task_name=task.name, unique_config=task.config, type='json')
#         # and inside of his store class he registers different functions to type keys
#         # e.g. 'json': self.save_json
#         #      'torch': self.save_torch
#         # and the general self.store.save() fn queries and calls the corresponding fn by type (e.g. 'json' -> save_json)
#
#
# class Evaluate(Task):
#     def __init__(self):
#         super().__init__()
#
#     def run(self, torch_model, dataset):
#         pass
#
#
# train = TaskSpec(train_fn, publishes=["torch_model", "dataset"])
#
# ############################################################
# ############################################################
#
#
# class ModelHandler(Handler):
#     def load():
#         pass
#
#     def save(task_name, task_config, name, obj):
#         pass
#
#
# class ResultsStore:
#     def __init__(self, handler_mapping: Optional[Dict[str, Type[Handler]]]):
#         self._handler_mapping = handler_mapping
#
#         # Idea: add a mapping from None to InMemoryHandler
#
#     @abstractmethod
#     def load_handler(self, task_name, unique_config, name) -> Handler:
#        # this is has to come from disk/db/pickle, and gives a saved handler instance
#        pass
#
#     @abstractmethod
#     def save_handler(self, task_name, unique_config, name, handler: Handler):
#         # he would save the meta data for loading back
#         pass
#
#     def get_results(self):
#         results = {}
#         for name in publishes:
#             # TBD: handle result not found
#             handler = self.load_handler(task_name, unique_config, item_name)
#             results[name] = handler.load()
#         return results
#
#     def save(self, task_name, task_config, name, obj, type="json", **kwargs):
#         handler = self._handler_mapping[type](kwargs)
#         handler.save(task_name, task_config, name, obj)
#         self.save_handler(task_name, task_config, handler)
#
#
# class InMemoryStore(ResultsStore):
#     def __init__(self):
#         self._memory_store = Manager().dict()
#
#     def load_handler(self, task_name, unique_config, name) -> Handler:
#        # this is has to come from disk/db/pickle, and gives a saved handler instance
#        pass
#
#     def save_handler(self, task_name, unique_config, handler: Handler):
#         # he would save the meta data for loading back
#         pass
#
#     def get_results(self, task_name: str, unique_config: Dict, publishes: List[str]) -> Optional[Dict]:
#         # if no store is used, default to get results from in-memory
#         if task_name not in self._memory_store:
#             return None
#
#         for task_sweep in self._memory_store[task_name]:
#             if task_sweep["config"] == unique_config:
#                 results = {}
#                 for item_name in publishes:
#                     try:
#                         results[item_name] = task_sweep['results'][item_name]
#                     except KeyError:
#                         raise KeyError(f'{item_name} not saved.')
#
#                 return results
#
#     def save(self, name: str, obj: Any, task_name: str, unique_config: Dict):
#         """ In-memory save function.
#         Adds individual object to in-memory store (multiprocessing manager dict).
#         """
#         if task_name not in self._memory_store:
#             self._memory_store[task_name] = []
#
#         existing_task_results = self._memory_store[task_name]
#         sweep_exists = False
#         for task_sweep in existing_task_results:
#             if task_sweep['config'] == unique_config:
#                 task_sweep['results'][name] = obj
#                 sweep_exists = True
#                 break
#
#         if not sweep_exists:
#             new_task_sweep = {'results': {name: obj},
#                               'config': unique_config}
#             existing_task_results.append(new_task_sweep)
#
#         self._memory_store[task_name] = existing_task_results
#
#
# class Train(Task):
#     publishes = ["torch_model", "dataset"]
#
#     def run():
#         results_store.save(task_name, config, "torch_model", model, "model")


###############################################################

# class Handler(ABC):
#     @abstractmethod
#     def load(self, name: str):
#         pass
#
#     def save(self, name: str, obj: Any):
#         pass
#
#
# class ResultsStore(ABC):
#     def __init__(self, handler_mapping: Optional[Dict[str, Handler]]):
#         self._handler_mapping = handler_mapping
#
#     @abstractmethod
#     def load_handler(self, name: str, task_name: str, unique_config: Dict) -> Handler:
#         # this is has to come from disk/db/pickle, and gives a saved handler instance
#         pass
#
#     @abstractmethod
#     def save_handler(self, name: str, task_name: str, unique_config: Dict, handler: Handler):
#         # he would save the meta data for loading back
#         pass
#
#     def get_results(self, task_name: str, unique_config: Dict, publishes: List[str]) -> Optional[Dict]:
#         results = {}
#         for name in publishes:
#             # TBD: handle result not found
#             handler = self.load_handler(name=name, task_name=task_name, unique_config=unique_config)
#             results[name] = handler.load(name=name)
#         return results
#
#     def save(self, name: str, obj: Any, task_name: str, task_config: Dict, type_: str, **kwargs):
#         handler_cls = self._handler_mapping[type_]
#         handler = handler_cls(task_name, task_config, kwargs)
#         handler.save(name=name, obj=obj)
#         self.save_handler(name=name, task_name=task_name, unique_config=task_config, handler=handler)
#
#
# class JsonHandler(Handler):
#
#     def __init__(self, task_name: str, task_config: Dict, kwargs: Dict):
#         self.kwargs = kwargs
#         self.run_dir = self._get_run_dir(task_name=task_name, unique_config=task_config)
#
#     def load(self, name: str) -> Dict:
#         obj = json.load(open(os.path.join(self.run_dir, f'{name}.json'), 'r'))
#         return obj
#
#     def save(self, name: str, obj: Dict):
#         json.dump(obj, open(os.path.join(self.run_dir, f'{name}.json'), 'w'))
#
#
# class TorchHandler(Handler):
#
#     def __init__(self, task_name: str, task_config: Dict, kwargs: Dict):
#         self.kwargs = kwargs
#         self.run_dir = self._get_run_dir(task_name=task_name, unique_config=task_config)
#
#     def load(self, name: str) -> nn.Module:
#         model_state_dict = torch.load(os.path.join(self.run_dir, f'{name}.pt'))
#         return model_state_dict
#
#     def save(self, name: str, obj: nn.Module):
#         torch.save(obj.state_dict(), os.path.join(self.run_dir, f'{name}.pt'))
#
#
# class LocalFileStore(ResultsStore):
#     handler_mapping = {'json': JsonHandler,
#                        'torch': TorchHandler}
#
#     def __init__(self, base_dir: str):
#         super().__init__(handler_mapping=self.handler_mapping)
#         self.base_dir = base_dir
#
#     def load_handler(self, name: str, task_name: str, unique_config: Dict) -> Handler:
#         # get run dir with task_name and unique_config
#         run_dir = self._get_run_dir(task_name=task_name, unique_config=unique_config)
#
#         # load handler
#         handler = pickle.load(open(os.path.join(run_dir, f"{name}_handler.p"), "rb"))
#
#         return handler
#
#     def save_handler(self, name: str, task_name: str, unique_config: Dict, handler: Handler):
#         # get run dir with task_name and unique_config
#         run_dir = self._get_run_dir(task_name=task_name, unique_config=unique_config)
#
#         # save handler
#         pickle.dump(handler, open(os.path.join(run_dir, f"{name}_handler.p"), "wb"))
#
#
#
# #################
# ################
#
#
# class ResultsStore(ABC):
#
#     @abstractmethod
#     def get_results(self, task_name: str, unique_config: Dict, publishes: List[str]) -> Optional[Dict]:
#         pass
#
#     @abstractmethod
#     def save(self, name: str, obj: Any, type_: str, task_name: str, unique_config: Dict, **kwargs):
#         pass
#
#
#
#
# class LocalFileStore(ResultsStore):
#     def __init__(self):
#         self._save_load_from_type = {'json': (self.save_json, self._load_json),
#                                      'torch': (self.save_torch, self._load_torch)}
#
#     def get_results(self, task_name: str, unique_config: Dict, publishes: List[str]) -> Optional[Dict]:
#         # get run dir with task_name and unique_config
#         run_dir = self._get_run_dir(task_name=task_name, unique_config=unique_config)
#
#         results = {}
#         for item_name in publishes:
#             # load load-info
#             try:
#                 load_info = pickle.load(open(run_dir + f"{item_name}_load_info.p", "wb"))
#             except FileNotFoundError:
#                 raise FileNotFoundError(f'{item_name} not saved.')
#
#             load_fn = load_info['load_fn']
#             kwargs = load_info['kwargs']
#
#             # load the saved object
#             obj = load_fn(name=item_name, run_dir=run_dir, **kwargs)
#
#             # store obj in results
#             results[item_name] = obj
#         return results
#
#     def save_json(self, name: str, obj: Dict, run_dir: str):
#         # save dict
#         json.dump(obj, open(os.path.join(run_dir, f'{name}.json'), "w"))
#
#     def _load_json(self, name: str, run_dir: str) -> Dict:
#         return json.load(open(os.path.join(run_dir, f'{name}.json'), "w"))
#
#     def save_torch(self, name: str, obj: nn.Module, run_dir: str):
#         # save model
#         torch.save(obj.state_dict(), os.path.join(run_dir, f'{name}.pt'))
#
#     def _load_torch(self, name: str, run_dir: str):
#         return torch.load(os.path.join(run_dir, f'{name}.pt'))
#
#     def save(self, name: str, obj: Any, type_: str, task_name: str, unique_config: Dict, **kwargs):
#         # get run dir with task_name and unique_config
#         run_dir = self._get_run_dir(task_name=task_name, unique_config=unique_config)
#
#         save_fn, load_fn = self._save_load_from_type[type_]
#
#         # save obj
#         save_fn(name=name, obj=obj, run_dir=run_dir, **kwargs)
#
#         # save load info
#         load_info = {'kwargs': kwargs,
#                      'load_fn': load_fn}
#         pickle.dump(load_info, open(os.path.join(run_dir, f'{name}_load_info.p'), 'wb'))


#################
#################


class ResultsStore(ABC):
    @abstractmethod
    def load(self, name: str, task_name: str, task_config: Dict) -> Optional[Any]:
        raise NotImplementedError

    @abstractmethod
    def save(self, name: str, obj: Any, type_: str, task_name: str, task_config: Dict, **kwargs):
        raise NotImplementedError

    def get_results(self, task_name: str, task_config: Dict, publishes: List[str]) -> Optional[Dict]:
        # here we loop over individual item names and call user provided self.load() to get individual item data
        results = {}
        for item_name in publishes:
            # load object
            obj = self.load(name=item_name, task_name=task_name, task_config=task_config)

            # store object in results
            results[item_name] = obj
        return results


class InMemoryStore(ResultsStore):
    def __init__(self):
        self._memory_store = Manager().dict()

    def load(self, name: str, task_name: str, task_config: Dict) -> Optional[Dict]:
        if task_name not in self._memory_store:
            return None

        for task_sweep in self._memory_store[task_name]:
            if task_sweep["config"] == task_config:
                try:
                    obj = task_sweep['results'][name]
                except KeyError:
                    raise KeyError(f'{name} not saved.')

                return obj

    def save(self, name: str, obj: Any, task_name: str, task_config: Dict, **kwargs):
        """ In-memory save function.
        Adds individual object to in-memory store (multiprocessing manager dict).
        """
        if task_name not in self._memory_store:
            self._memory_store[task_name] = []

        existing_task_results = self._memory_store[task_name]
        sweep_exists = False
        for task_sweep in existing_task_results:
            if task_sweep['config'] == task_config:
                task_sweep['results'][name] = obj
                sweep_exists = True
                break

        if not sweep_exists:
            new_task_sweep = {'results': {name: obj},
                              'config': task_config}
            existing_task_results.append(new_task_sweep)

        self._memory_store[task_name] = existing_task_results


class LocalFileStore(ResultsStore):
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self._save_load_from_type = {'json': (self._save_json, self._load_json),
                                     'torch': (self._save_torch, self._load_torch)}

    @staticmethod
    def _save_json(name: str, obj: Dict, run_dir: str):
        # save dict
        json.dump(obj, open(os.path.join(run_dir, f'{name}.json'), "w"))

    @staticmethod
    def _load_json(name: str, run_dir: str) -> Dict:
        return json.load(open(os.path.join(run_dir, f'{name}.json'), "w"))

    @staticmethod
    def _save_torch(name: str, obj: nn.Module, run_dir: str):
        # save model
        torch.save(obj.state_dict(), os.path.join(run_dir, f'{name}.pt'))

    @staticmethod
    def _load_torch(name: str, run_dir: str):
        return torch.load(os.path.join(run_dir, f'{name}.pt'))

    def save(self, name: str, obj: Any, type_: str, task_name: str, task_config: Dict, **kwargs):
        task_dir = os.path.join(self.base_dir, task_name)

        # try to get existing run dir
        run_dir = LocalFileStore._get_run_dir(task_dir=task_dir, task_config=task_config)

        # create new run dir if run dir did not exist
        if run_dir is None:
            run_dir = LocalFileStore._make_run_dir(task_dir=task_dir)

        # get save and load function for type
        save_fn, load_fn = self._save_load_from_type[type_]

        # save object
        save_fn(name=name, obj=obj, run_dir=run_dir, **kwargs)

        # save load info
        load_info = {'kwargs': kwargs,
                     'load_fn': load_fn}
        pickle.dump(load_info, open(os.path.join(run_dir, f'{name}_load_info.p'), 'wb'))

    def load(self, name: str, task_name: str, task_config: Dict) -> Optional[Any]:
        task_dir = os.path.join(self.base_dir, task_name)

        # try to get existing run dir
        run_dir = LocalFileStore._get_run_dir(task_dir=task_dir, task_config=task_config)
        if run_dir is None:
            return None

        # get load information from run dir
        try:
            load_info = pickle.load(open(run_dir + f"{name}_load_info.p", "wb"))
        except FileNotFoundError:
            raise FileNotFoundError(f'{name} not saved.')

        # unpack load info
        load_fn = load_info['load_fn']
        kwargs = load_info['kwargs']

        # load the saved object from run dir
        obj = load_fn(name=name, run_dir=run_dir, **kwargs)
        return obj

    @staticmethod
    def _scan_task_dir(task_dir: str) -> List[str]:
        os.makedirs(task_dir, exist_ok=True)
        exist_run_dirs = [os.path.join(task_dir, d.name)
                          for d in os.scandir(task_dir)
                          if d.is_dir() and d.name.isdigit()]
        return exist_run_dirs

    @staticmethod
    def _get_run_dir(task_dir: str, task_config: Dict) -> Optional[str]:
        exist_run_dirs = LocalFileStore._scan_task_dir(task_dir=task_dir)
        for exist_run_dir in exist_run_dirs:
            try:
                exist_config = json.load(open(os.path.join(exist_run_dir, 'config.json'), 'r'))
            except FileNotFoundError:
                continue
            if task_config == exist_config:
                return exist_run_dir
        return None

    @staticmethod
    def _make_run_dir(task_dir: str) -> str:
        exist_run_dirs = LocalFileStore._scan_task_dir(task_dir=task_dir)
        new_id = max([int(os.path.split(d)[-1]) for d in exist_run_dirs]) + 1 if exist_run_dirs else 0
        new_run_dir = os.path.join(task_dir, f'{str(new_id).zfill(3)}')
        os.makedirs(new_run_dir, exist_ok=True)
        return new_run_dir


# 1. Simplify fluidml imports, add extra __init__ on top level for simle imports
# 2. Instead of calling self.results_store.save(...), add save method directly to task as
#    self.save = self.results_store.save
# 3. Dynamically add task_name and task_unique_config to results_store as attributes
#    self.results_store.unique_config = self.unique_config
#    self.results_store.task_name = self.task_name
#    -> self.results_store.save(obj, name, type, **kwargs) instead of
#       self.results_store.save(obj, name, type, task_name, task_unique_config, **kwargs)
