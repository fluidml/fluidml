import pickle
from typing import Optional, Dict

import mongoengine as me

from fluidml.storage import ResultsStore


def connection(func):
    """ Decorator to handle connecting and disconnecting to/from db """
    def wrapper_connect_disconnect_db(self, *args, **kwargs):
        me.connect(db=self._db, host=self._host)
        result = func(self, *args, **kwargs)
        me.disconnect()
        return result

    # support for calling method without decorator logic
    wrapper_connect_disconnect_db.no_connection = func
    return wrapper_connect_disconnect_db


class MongoDBStore(ResultsStore):
    def __init__(self,
                 db: str,
                 collection_name: Optional[str] = None,
                 host: Optional[str] = None):
        self._host = host
        self._db = db
        self._collection_name = collection_name

    @connection
    def get_results(self, task_name: str, unique_config: Dict) -> Optional[Dict]:
        """ Returns the latest results if available """
        task_result_cls = self._get_task_result_class()
        try:
            result = task_result_cls.objects(name=task_name).get(unique_config=unique_config)
            pass
        except me.DoesNotExist:
            result = None
        if result is not None:
            result = pickle.loads(result.results.read())
        return result

    @connection
    def update_results(self, task_name: str, unique_config: Dict, results: Dict):
        """ Overwrite existing results """
        task_result_cls = self._get_task_result_class()
        try:
            result = task_result_cls.objects(name=task_name).get(unique_config=unique_config)
            pass
        except me.DoesNotExist:
            result = None
        if result is not None:
            result.delete()
        task_result = task_result_cls(name=task_name,
                                      unique_config=unique_config,
                                      results=pickle.dumps(results))
        task_result.save()

    @connection
    def save_results(self, task_name: str, unique_config: Dict, results: Dict):
        """ Save new results """
        task_result_cls = self._get_task_result_class()
        task_result = task_result_cls(name=task_name,
                                      unique_config=unique_config,
                                      results=pickle.dumps(results))
        task_result.save()

    def _get_task_result_class(self):
        # Hack to set the collection name dynamically from user input
        # Default is the document class name lower-cased, here: "task_result"
        class TaskResult(me.Document):
            name = me.StringField()
            unique_config = me.DictField()
            results = me.FileField()
            if self._collection_name is not None:
                meta = {'collection': self._collection_name}
        return TaskResult


def main():
    store = MongoDBStore("test")
    task_config = {"param_a": 23, "param_b": 55}
    task_results = {"Result": 5}
    task_name = "task_1"
    store.update_results(task_name, task_config, task_results)
    result = store.get_results(task_name, task_config)
    print(result)


if __name__ == "__main__":
    main()
