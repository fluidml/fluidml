from busy_bee import Swarm, Task, Resource
from typing import Dict, Any
from torchnlp.datasets import trec_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from flair.data import Sentence
from flair.embeddings import WordEmbeddings, DocumentPoolEmbeddings
import numpy as np


def results_available(results, task_name, value) -> bool:
    return results.get(task_name, None) is not None and results[task_name].get(value, None) is not None


class DatasetFetchTask(Task):
    def __init__(self, id_: int):
        super().__init__(id_, "dataset")

    def run(self, results: Dict[str, Any], resource: Resource):
        dataset = trec_dataset(train=True)
        sentences = [sample["text"] for sample in dataset]
        labels = [sample["label"] for sample in dataset]
        task_results = {
            "sentences": sentences,
            "labels": labels
        }
        return task_results


class PreProcessTask(Task):
    def __init__(self, id_: int):
        super().__init__(id_, "pre_process")

    def run(self, results: Dict[str, Any], resource: Resource):
        assert results_available(results, "dataset", "sentences"), "Sentences must be present"
        pre_processed_sentences = [sentence for sentence in results["dataset"]["sentences"]]
        task_results = {
            "sentences": pre_processed_sentences,
        }
        return task_results


class TFIDFFeaturizeTask(Task):
    def __init__(self, id_: int):
        super().__init__(id_, "tfidf_featurize")

    def run(self, results: Dict[str, Any], resource: Resource):
        assert results_available(results, "pre_process", "sentences"), "Sentences must be present"
        tfidf = TfidfVectorizer()
        tfidf_vectors = tfidf.fit_transform(results["pre_process"]["sentences"]).toarray()
        task_results = {
            "vectors": tfidf_vectors
        }
        return task_results


class GloveFeaturizeTask(Task):
    def __init__(self, id_: int):
        super().__init__(id_, "glove_featurize")

    def run(self, results: Dict[str, Any], resource: Resource):
        assert results_available(results, "pre_process", "sentences"), "Sentences must be present"
        sentences = [Sentence(sent) for sent in results["pre_process"]["sentences"]]
        embedder = DocumentPoolEmbeddings([WordEmbeddings("glove")])
        embedder.embed(sentences)
        glove_vectors = [sent.embedding.cpu().numpy() for sent in sentences]
        glove_vectors = np.array(glove_vectors).reshape(len(glove_vectors), -1)
        task_results = {
            "vectors": glove_vectors
        }
        return task_results


class TrainTask(Task):
    def __init__(self, id_: int):
        super().__init__(id_, "train")

    def run(self, results: Dict[str, Any], resource: Resource):
        assert results_available(results, "tfidf_featurize", "vectors"), "TF-IDF Vectors must be present"
        assert results_available(results, "glove_featurize", "vectors"), "Glove Vectors must be present"
        assert results_available(results, "dataset", "labels"), "Labels must be present"
        model = LogisticRegression(max_iter=50)
        stacked_vectors = np.hstack((results["tfidf_featurize"]["vectors"], results["glove_featurize"]["vectors"]))
        model.fit(stacked_vectors, results["dataset"]["labels"])
        task_results = {
            "model": model
        }
        return task_results


class EvaluateTask(Task):
    def __init__(self, id_: int):
        super().__init__(id_, "evaluate_task")

    def run(self, results: Dict[str, Any], resource: Resource):
        assert results_available(results, "tfidf_featurize", "vectors"), "TF-IDF Vectors must be present"
        assert results_available(results, "glove_featurize", "vectors"), "Glove Vectors must be present"
        assert results_available(results, "dataset", "labels"), "Labels must be present"
        assert results_available(results, "train", "model"), "Trained model must be present"
        stacked_vectors = np.hstack((results["tfidf_featurize"]["vectors"], results["glove_featurize"]["vectors"]))
        predictions = results["train"]["model"].predict(stacked_vectors)
        report = classification_report(results["dataset"]["labels"], predictions)
        task_results = {
            "classification_report": report
        }
        return task_results


def main():

    # create all tasks
    dataset_fetch_task = DatasetFetchTask(1)
    pre_process_task = PreProcessTask(2)
    featurize_task_1 = TFIDFFeaturizeTask(3)
    featurize_task_2 = GloveFeaturizeTask(4)
    train_task = TrainTask(5)
    evaluate_task = EvaluateTask(6)

    # dependencies between tasks
    pre_process_task.requires([dataset_fetch_task])
    featurize_task_1.requires([pre_process_task])
    featurize_task_2.requires([pre_process_task])
    train_task.requires([dataset_fetch_task, featurize_task_1, featurize_task_2])
    evaluate_task.requires([dataset_fetch_task, featurize_task_1, featurize_task_2, train_task])

    # all tasks
    tasks = [dataset_fetch_task,
             pre_process_task,
             featurize_task_1, featurize_task_2,
             train_task,
             evaluate_task]

    with Swarm(n_bees=3, refresh_every=5) as swarm:
        results = swarm.work(tasks)
    print(results[6]["classification_report"])


if __name__ == "__main__":
    main()
