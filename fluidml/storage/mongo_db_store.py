from typing import Optional, Dict
from pymongo import MongoClient
from bson import Binary
import pickle

from fluidml.storage import ResultsStore


class MongoDBStore(ResultsStore):
    def __init__(self, uri: str, db: str, collection_name: str):
        self._uri = uri
        self._db = db
        self._collection_name = collection_name
        MongoDBStore._create_collection(self._get_client(), self._collection_name)

    @staticmethod
    def _create_collection(client: MongoClient, collection_name: str):
        pass

    def _get_client(self):
        client = MongoClient(self._uri)
        return client

    def get_results(self, task_name: str, unique_config: Dict) -> Optional[Dict]:
        """
        Returns the latest results
        """
        client = self._get_client()
        collection = client[self._db][self._collection_name]
        result = collection.find(filter={"name": task_name, "unique_config": unique_config})
        fetched_results = None
        if result.count() > 0:
            result = [item for item in result]
            result = sorted(result, key=lambda item: item["_id"].generation_time, reverse=True)
            fetched_results = pickle.loads(result[0]["results"])
        client.close()
        return fetched_results

    def update_results(self, task_name: str, unique_config: Dict, results: Dict) -> str:
        self.save_results(task_name, unique_config, results)

    def save_results(self, task_name: str, unique_config: Dict, results: Dict) -> str:
        client = self._get_client()
        collection = client[self._db][self._collection_name]
        document = {
            "name": task_name,
            "unique_config": unique_config,
            "results": Binary(pickle.dumps(results))
        }
        collection.insert_one(document)
        client.close()


if __name__ == "__main__":
    store = MongoDBStore("mongodb://localhost:27017/", "test", "results_collection")
    task_config = {"param_a": 23, "param_b": 55}
    task_results = "Result"
    task_name = "task_1"
    store.save_results(task_name, task_config, task_results)
    result = store.get_results(task_name, task_config)
    print(result)
