from typing import Dict, Any

from datasets import load_dataset
from flair.data import Sentence
from flair.embeddings import WordEmbeddings, DocumentPoolEmbeddings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from fluidml.common.task import Task, Resource
from fluidml.swarm import Swarm
from fluidml.flow import Flow
from fluidml.flow.task_spec import TaskSpec
from fluidml.storage.mongo_db_store import MongoDBStore


def results_available(results, task_name, value) -> bool:
    return results.get(task_name, None) is not None and \
           results[task_name].get("result", None) is not None and  \
           results[task_name]["result"].get(value, None) is not None


class DatasetFetchTask(Task):
    def __init__(self, name: str, id_: int):
        super().__init__(name, id_)

    def run(self, results: Dict[str, Any], resource: Resource):
        dataset = load_dataset("reuters21578", "ModApte")
        sentences = []
        labels = []
        for item in dataset["train"]:
            if len(item["topics"]) > 0:
                sentences.append(item["text"])
                labels.append(item["topics"][0])
        task_results = {
            "sentences": sentences,
            "labels": labels
        }
        return task_results


class PreProcessTask(Task):
    def __init__(self, name: str, id_: int):
        super().__init__(name, id_)

    def run(self, results: Dict[str, Any], resource: Resource):
        assert results_available(results, "dataset", "sentences"), "Sentences must be present"
        pre_processed_sentences = [sentence for sentence in results["dataset"]["result"]["sentences"]]
        task_results = {
            "sentences": pre_processed_sentences,
        }
        return task_results


class TFIDFFeaturizeTask(Task):
    def __init__(self, name: str, id_: int):
        super().__init__(name, id_)

    def run(self, results: Dict[str, Any], resource: Resource):
        assert results_available(results, "pre_process", "sentences"), "Sentences must be present"
        tfidf = TfidfVectorizer()
        tfidf_vectors = tfidf.fit_transform(results["pre_process"]["result"]["sentences"]).toarray()
        task_results = {
            "vectors": tfidf_vectors
        }
        return task_results


class GloveFeaturizeTask(Task):
    def __init__(self, name: str, id_: int):
        super().__init__(name, id_)

    def run(self, results: Dict[str, Any], resource: Resource):
        assert results_available(results, "pre_process", "sentences"), "Sentences must be present"
        sentences = [Sentence(sent) for sent in results["pre_process"]["result"]["sentences"]]
        embedder = DocumentPoolEmbeddings([WordEmbeddings("glove")])
        embedder.embed(sentences)
        glove_vectors = [sent.embedding.cpu().numpy() for sent in sentences]
        glove_vectors = np.array(glove_vectors).reshape(len(glove_vectors), -1)
        task_results = {
            "vectors": glove_vectors
        }
        return task_results


class TrainTask(Task):
    def __init__(self, name: str, id_: int):
        super().__init__(name, id_)

    def run(self, results: Dict[str, Any], resource: Resource):
        assert results_available(results, "tfidf_featurize", "vectors"), "TF-IDF Vectors must be present"
        assert results_available(results, "glove_featurize", "vectors"), "Glove Vectors must be present"
        assert results_available(results, "dataset", "labels"), "Labels must be present"
        model = LogisticRegression(max_iter=50)
        stacked_vectors = np.hstack((results["tfidf_featurize"]["result"]["vectors"],
                                     results["glove_featurize"]["result"]["vectors"]))
        model.fit(stacked_vectors, results["dataset"]["result"]["labels"])
        task_results = {
            "model": model,
            "vectors": stacked_vectors,
            "labels": results["dataset"]["result"]["labels"],
        }
        return task_results


class EvaluateTask(Task):
    def __init__(self, name: str, id_: int):
        super().__init__(name, id_)

    def run(self, results: Dict[str, Any], resource: Resource):
        assert results_available(results, "train", "model"), "Trained model must be present"
        assert results_available(results, "train", "vectors"), "Vectors must be present"
        assert results_available(results, "train", "labels"), "labels must be present"
        predictions = results["train"]["result"]["model"].predict(results["train"]["result"]["vectors"])
        report = classification_report(results["train"]["result"]["labels"], predictions, output_dict=True)
        task_results = {
            "classification_report": report
        }
        return task_results


def main():

    # create all task specs
    dataset_fetch_task = TaskSpec(task=DatasetFetchTask, name="dataset")
    pre_process_task = TaskSpec(task=PreProcessTask, name="pre_process")
    featurize_task_1 = TaskSpec(task=GloveFeaturizeTask, name="glove_featurize")
    featurize_task_2 = TaskSpec(task=TFIDFFeaturizeTask, name="tfidf_featurize")
    train_task = TaskSpec(task=TrainTask, name="train")
    evaluate_task = TaskSpec(task=EvaluateTask, name="evaluate")

    # dependencies between tasks
    pre_process_task.requires([dataset_fetch_task])
    featurize_task_1.requires([pre_process_task])
    featurize_task_2.requires([pre_process_task])
    train_task.requires([dataset_fetch_task, featurize_task_1, featurize_task_2])
    evaluate_task.requires([train_task])

    # all tasks
    tasks = [dataset_fetch_task,
             pre_process_task,
             featurize_task_1, featurize_task_2,
             train_task,
             evaluate_task]

    # mongo results store
    results_store = MongoDBStore("test2")

    with Swarm(n_dolphins=2,
               refresh_every=10,
               return_results=True,
               results_store=results_store) as swarm:
        flow = Flow(swarm=swarm)
        results = flow.run(tasks)
    print(results["evaluate"])


if __name__ == "__main__":
    main()
