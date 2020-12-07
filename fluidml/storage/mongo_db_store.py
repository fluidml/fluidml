# from bson import Binary
import pickle
from typing import Optional, Dict

import mongoengine as me
# from pymongo import MongoClient

from fluidml.storage import ResultsStore


class MongoDBStore(ResultsStore):
    def __init__(self,
                 db: str,
                 collection_name: Optional[str] = None,
                 host: Optional[str] = None):
        self._host = host
        self._db = db
        self._collection_name = collection_name

    def get_results(self, task_name: str, unique_config: Dict) -> Optional[Dict]:
        """
        Returns the latest results
        """
        me.connect(db=self._db, host=self._host)
        task_result_cls = self._get_task_result_class()
        result = task_result_cls.objects(name=task_name, unique_config=unique_config)

        fetched_results = None
        if result.count() > 0:
            result = [item for item in result]
            result = sorted(result, key=lambda item: item.id.generation_time, reverse=True)
            fetched_results = pickle.loads(result[0].results.read())
        me.disconnect()
        return fetched_results

    def update_results(self, task_name: str, unique_config: Dict, results: Dict):
        self.save_results(task_name, unique_config, results)

    def save_results(self, task_name: str, unique_config: Dict, results: Dict):
        me.connect(db=self._db, host=self._host)
        task_result_cls = self._get_task_result_class()
        task_result = task_result_cls(name=task_name,
                                      unique_config=unique_config,
                                      results=pickle.dumps(results))
        task_result.save()
        me.disconnect()

    def _get_task_result_class(self):
        class TaskResult(me.Document):
            name = me.StringField()
            unique_config = me.DictField()
            results = me.FileField()
            if self._collection_name is not None:
                meta = {'collection': self._collection_name}
        return TaskResult


# class MongoDBStore(ResultsStore):
#     def __init__(self, uri: str, db: str, collection_name: str):
#         self._uri = uri
#         self._db = db
#         self._collection_name = collection_name
#
#     def _get_client(self):
#         client = MongoClient(self._uri)
#         return client
#
#     def get_results(self, task_name: str, unique_config: Dict) -> Optional[Dict]:
#         """
#         Returns the latest results
#         """
#         client = self._get_client()
#         collection = client[self._db][self._collection_name]
#         result = collection.find(filter={"name": task_name, "unique_config": unique_config})
#         fetched_results = None
#         if len(list(result)) > 0:
#             result = [item for item in result]
#             result = sorted(result, key=lambda item: item["_id"].generation_time, reverse=True)
#             fetched_results = pickle.loads(result[0]["results"])
#         client.close()
#         return fetched_results
#
#     def update_results(self, task_name: str, unique_config: Dict, results: Dict):
#         self.save_results(task_name, unique_config, results)
#
#     def save_results(self, task_name: str, unique_config: Dict, results: Dict):
#         client = self._get_client()
#         collection = client[self._db][self._collection_name]
#         document = {
#             "name": task_name,
#             "unique_config": unique_config,
#             "results": Binary(pickle.dumps(results))
#         }
#         collection.insert_one(document)
#         client.close()


def main():
    store = MongoDBStore("test")
    task_config = {"param_a": 23, "param_b": 55}
    task_results = {"Result": 5}
    task_name = "task_1"
    store.save_results(task_name, task_config, task_results)
    result = store.get_results(task_name, task_config)
    print(result)


if __name__ == "__main__":
    main()
