from typing import Dict, Any, Callable
import json


class ResultItem:
    name: str                                       # a unique name for the result (eg. raw_dataset)
    data: Any                                       # any data which is to be stored
    save_fn: Callable[str, Dict, ResultItem]        # a callable that is specified by user how to save the data
    load_fn: Callable[str, Dict, str]               # a callable that is specified by user how to load the data back
    load_kwargs: Optional[Dict]
    save_kwargs: Optional[Dict]


class LocalFileStore:
    def save_json_data(self, task_name: str, unique_config: Dict, result_item: ResultItem):
        # get run dir
        json_path = run_dir + result_item.name

        # dump
        json.dump(result_item.data, open(json_path, "w"))

        # dump meta data
        meta_data = {"name": result_item.name, "load_fn": result_item.load_fn}
        pickle.dump(meta_data, open(run_dir + f"{result_item.name}_meta.p", "wb"))

    def load_json_data(self, task_name, task_config, name) -> Dict:
        # load the meta data
        json_path = run_dir + name

        data = json.load(open(json_path, "w"))
        return data


class MongoDBStore:
    pass



class ResultsStore:

    # TBD: include manager dict to have in memory storage

    def get_meta_item(self, name: str) -> Dict:
        pass

    def get_results(self, task_name: str, unique_config: Dict) -> Optional[Dict]:
        # here we need to figure how to load all the data

        results_dict = {}
        for item_name in publishes:
            # we need a function from user that gives meta data

            # get the meta data with task name, config and item name

            # call the load fn

            # get the data using load fn
            # collect in results dict
        
        return results_dict

    def save(self, task_name: str, unique_config: Dict, result_items: List[ResultItem]):
        """ Method to save new results """

        for item in result_items:
            if item.save_fn is None:
                # tbd - add to manager dict
            else:
                result_item.save_fn(result_item.data, **result_item.save_kwargs)


class Train:

    publishes = ["torch_model", "dataset"]
    self.store = store

    def run(self):
        string_result = ResultItem("log_txt", {"something"}, self.store.save_json_data, self.store.load_json_data)
        self.results_store.save(string_result)


class Evaluate(Task):
    def __init__(self):
        super().__init__()

    def run(self, torch_model, dataset):
        pass



train = TaskSpec(train_fn, publishes=["torch_model", "dataset"])