import pickle
from typing import Optional, Dict, Any

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


class ResultObject(me.EmbeddedDocument):
    obj = me.FileField()


class MongoDBStore(ResultsStore):
    def __init__(self,
                 db: str,
                 collection_name: Optional[str] = None,
                 host: Optional[str] = None):
        self._host = host
        self._db = db
        self._collection_name = collection_name

    @connection
    def load(self, name: str, task_name: str, task_unique_config: Dict) -> Optional[Any]:
        """ Query method to load an object based on its name, task_name and task_config if it exists """
        task_result_cls = self._get_task_result_class()
        try:
            task_result = task_result_cls.objects(name=task_name).get(unique_config=task_unique_config)
        except me.DoesNotExist:
            return None
        try:
            result_obj = task_result.results[name]
            obj = pickle.loads(result_obj.obj.read())
        except KeyError:
            raise KeyError(f'Object "{name}" does not exist.')
        return obj

    @connection
    def save(self, obj: Any, name: str, type_: str, task_name: str, task_unique_config: Dict, **kwargs):
        """ Method to save/update any artifact """
        task_result_cls = self._get_task_result_class()
        try:
            task_result = task_result_cls.objects(name=task_name).get(unique_config=task_unique_config)
        except me.DoesNotExist:
            task_result = task_result_cls(name=task_name,
                                          unique_config=task_unique_config)

        result_obj = ResultObject(obj=pickle.dumps(obj))
        task_result.results[name] = result_obj
        task_result.save()

    def _get_task_result_class(self):
        # Hack to set the collection name dynamically from user input
        # Default is the document class name lower-cased, here: "task_result"
        class TaskResult(me.Document):
            name = me.StringField()
            unique_config = me.DictField()
            results = me.DictField(me.EmbeddedDocumentField(ResultObject))
            if self._collection_name is not None:
                meta = {'collection': self._collection_name}
        return TaskResult


def main():
    store = MongoDBStore("test")

    task_config = {"param_a": 23, "param_b": 55}
    obj1 = {"Result": 5}
    obj2 = 'hallo'
    task_name = "task_1"

    store.save(obj=obj1, name='obj3', type_='a', task_name=task_name, task_unique_config=task_config)
    store.save(obj=obj2, name='obj2', type_='b', task_name=task_name, task_unique_config=task_config)
    result1 = store.load(name='obj1', task_name=task_name, task_unique_config=task_config)
    result2 = store.load(name='obj2', task_name=task_name, task_unique_config=task_config)
    print(result1)
    print(result2)


if __name__ == "__main__":
    main()
