import functools
import logging
import pickle
from typing import Any, Dict, Optional

import mongoengine as me

from fluidml.storage.base import ResultsStore, StoreContext

logger = logging.getLogger(__name__)


def connection(func):
    """Decorator to handle connecting and disconnecting to/from db"""

    @functools.wraps(func)
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
    """A mongo db result store implementation.

    Args:
        db: The database name to be used.
        collection_name: The name of the collection.
        host: The host name of the :program: `mongod` instance.
    """

    def __init__(self, db: str, collection_name: Optional[str] = None, host: Optional[str] = None):
        super().__init__()

        self._host = host
        self._db = db
        self._collection_name = collection_name

    @connection
    def load(
        self,
        name: str,
        task_name: str,
        task_unique_config: Dict,
        lazy: bool = False,
    ) -> Optional[Any]:
        """Query method to load an object based on its name, task_name and task_config if it exists"""
        task_result_cls = self._get_task_result_class()
        # try to get query run document based on task name and task unique config
        try:
            task_result = task_result_cls.objects(name=task_name).get(unique_config__lte=task_unique_config)
        except me.DoesNotExist:
            return None
        # try to query obj from results DictField
        try:
            result_obj = task_result.results[name]
            obj = pickle.loads(result_obj.obj.read())
        except KeyError:
            logger.warning(f'"{name}" could not be found in store.')
            return None
        return obj

    @connection
    def save(
        self,
        obj: Any,
        name: str,
        type_: str,
        task_name: str,
        task_unique_config: Dict,
        **kwargs,
    ):
        """Method to save/update any artifact"""
        task_result_cls = self._get_task_result_class()
        # try to get query run document based on task name and task unique config
        try:
            task_result = task_result_cls.objects(name=task_name).get(unique_config=task_unique_config)
        except me.DoesNotExist:
            # create new document if query was not successful
            task_result = task_result_cls(name=task_name, unique_config=task_unique_config)

        # store object in document and save the document
        result_obj = ResultObject(obj=pickle.dumps(obj))
        task_result.results[name] = result_obj
        task_result.save()

    @connection
    def delete(self, name: str, task_name: str, task_unique_config: Dict):
        """Query method to delete an object based on its name, task_name and task_config if it exists"""
        task_result_cls = self._get_task_result_class()
        # try to get query run document based on task name and task unique config
        try:
            task_result = task_result_cls.objects(name=task_name).get(unique_config__lte=task_unique_config)
        except me.DoesNotExist:
            logger.warning(
                f'"{name}" could not be deleted. '
                f'No document for task "{task_name}" and the provided unique_config exists.'
            )
            return None
        # try to delete obj from results DictField
        try:
            del task_result.results[name]
            task_result.save()
        except KeyError:
            logger.warning(f'"{name}" could not be deleted from store since it was not found.')

    @connection
    def delete_run(self, task_name: str, task_unique_config: Dict):
        """Query method to delete a run document based on its task_name and task_config if it exists"""
        task_result_cls = self._get_task_result_class()
        # try to get query run document based on task name and task unique config
        try:
            task_result = task_result_cls.objects(name=task_name).get(unique_config__lte=task_unique_config)
        except me.DoesNotExist:
            logger.warning(f'No document for task "{task_name}" and the provided unique_config exists.')
            return None

        # delete document
        task_result.delete()

    def get_context(self, task_name: str, task_unique_config: Dict) -> StoreContext:
        pass

    def _get_task_result_class(self):
        # Hack to set the collection name dynamically from user input
        # Default is the document class name lower-cased, here: "task_result"
        class TaskResult(me.DynamicDocument):
            name = me.StringField()
            unique_config = me.DictField()
            results = me.DictField(me.EmbeddedDocumentField(ResultObject))
            if self._collection_name is not None:
                meta = {"collection": self._collection_name}

        return TaskResult


def main():
    store = MongoDBStore("test")

    task_1_config = {"param_a": 23, "param_b": 55}
    task_2_config = {"param_a": 23, "param_b": 55, "param_c": 88}
    obj1 = {"Result": 5}
    obj2 = "hallo"
    task_1 = "task_1"
    task_2 = "task_2"

    store.save(
        obj=obj1,
        name="obj1",
        type_="a",
        task_name=task_1,
        task_unique_config=task_1_config,
    )
    store.save(
        obj=obj2,
        name="obj2",
        type_="b",
        task_name=task_2,
        task_unique_config=task_2_config,
    )
    # store.delete(name='obj2', task_name=task_name, task_unique_config=task_config)
    result1 = store.load(name="obj1", task_name=task_1, task_unique_config=task_1_config)
    result2 = store.load(name="obj2", task_name=task_2, task_unique_config=task_2_config)
    result3 = store.load(name="obj1", task_name=task_1, task_unique_config=task_2_config)
    print(result1)
    print(result2)
    print(result3)


if __name__ == "__main__":
    main()
