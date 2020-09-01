from busy_bee import Swarm, Task
from typing import Dict, Any
from torchnlp.datasets import trec_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from flair.data import Sentence
from flair.embeddings import WordEmbeddings, DocumentPoolEmbeddings
import numpy as np


class DatasetFetchTask(Task):
    def __init__(self, id_: int):
        super().__init__(id_, "DatasetFetchTask")

    def run(self, results: Dict[str, Any]):
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
        super().__init__(id_, "PreProcessTask")

    def run(self, results: Dict[str, Any]):
        assert "sentences" in results.keys(), "Sentences must be present"
        pre_processed_sentences = [sentence for sentence in results["sentences"]]
        task_results = {
            "processed_sentences": pre_processed_sentences,
        }
        return task_results


class TFIDFFeaturizeTask(Task):
    def __init__(self, id_: int):
        super().__init__(id_, "TFIDF-FeaturizeTask")

    def run(self, results: Dict[str, Any]):
        assert "processed_sentences" in results.keys(), "Sentences must be present"
        tfidf = TfidfVectorizer()
        tfidf_vectors = tfidf.fit_transform(results["processed_sentences"]).toarray()
        task_results = {
            "tfidf_vectors": tfidf_vectors
        }
        return task_results


class GloveFeaturizeTask(Task):
    def __init__(self, id_: int):
        super().__init__(id_, "Glove-FeaturizeTask")

    def run(self, results: Dict[str, Any]):
        assert "processed_sentences" in results.keys(), "Sentences must be present"
        sentences = [Sentence(sent) for sent in results["processed_sentences"]]
        embedder = DocumentPoolEmbeddings([WordEmbeddings("glove")])
        embedder.embed(sentences)
        glove_vectors = [sent.embedding.cpu().numpy() for sent in sentences]
        glove_vectors = np.array(glove_vectors).reshape(len(glove_vectors), -1)
        task_results = {
            "glove_vectors": glove_vectors
        }
        return task_results


class TrainTask(Task):
    def __init__(self, id_: int):
        super().__init__(id_, "TrainTask")

    def run(self, results: Dict[str, Any]):
        assert "tfidf_vectors" in results.keys(), "TF-IDF Vectors must be present"
        assert "glove_vectors" in results.keys(), "Glove Vectors must be present"
        assert "labels" in results.keys(), "Labels must be present"
        model = LogisticRegression(max_iter=50)
        stacked_vectors = np.hstack((results["tfidf_vectors"], results["glove_vectors"]))
        model.fit(stacked_vectors, results["labels"])
        task_results = {
            "model": model
        }
        return task_results


class EvaluateTask(Task):
    def __init__(self, id_: int):
        super().__init__(id_, "EvaluateTask")

    def run(self, results: Dict[str, Any]):
        assert "tfidf_vectors" in results.keys(), "TF-IDF Vectors must be present"
        assert "glove_vectors" in results.keys(), "Glove Vectors must be present"
        assert "labels" in results.keys(), "Labels must be present"
        assert "model" in results.keys(), "Trained model must be present"
        stacked_vectors = np.hstack((results["tfidf_vectors"], results["glove_vectors"]))
        predictions = results["model"].predict(stacked_vectors)
        report = classification_report(results["labels"], predictions)
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
    tasks = [dataset_fetch_task, pre_process_task, featurize_task_1, featurize_task_2, train_task, evaluate_task]

    with Swarm(n_bees=3, refresh_every=5) as swarm:
        results = swarm.work(tasks)
    print(results[6]["classification_report"])


if __name__ == "__main__":
    main()
