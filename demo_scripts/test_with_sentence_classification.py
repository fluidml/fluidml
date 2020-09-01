from busy_bee import Swarm, Task
from typing import List, Dict, Any
from torchnlp.datasets import trec_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from flair.data import Sentence
from flair.embeddings import WordEmbeddings, DocumentPoolEmbeddings


class DatasetFetchTask(Task):
    def __init__(self, id_: int, pre_task_ids: List[int], post_task_ids: List[int]):
        super().__init__(id_, pre_task_ids, post_task_ids)

    def run(self, results: Dict[str, Any]):
        # list of sentences
        dataset = trec_dataset(train=True)

        # list of sentences
        sentences = [sample["text"] for sample in dataset]
        labels = [sample["label"] for sample in dataset]

        # return results
        task_results = {
            "sentences": sentences,
            "labels": labels
        }
        return task_results


class PreProcessTask(Task):
    def __init__(self, id_: int, pre_task_ids: List[int], post_task_ids: List[int]):
        super().__init__(id_, pre_task_ids, post_task_ids)

    def run(self, results: Dict[str, Any]):
        assert "sentences" in results.keys(), "Sentences must be present"

        # pre-process sentences
        pre_processed_sentences = [sentence for sentence in results["sentences"]]

        # return results
        task_results = {
            "processed_sentences": pre_processed_sentences,
        }
        return task_results


class FeaturizeTask(Task):
    def __init__(self, id_: int, pre_task_ids: List[int], post_task_ids: List[int]):
        super().__init__(id_, pre_task_ids, post_task_ids)

    def run(self, results: Dict[str, Any]):
        assert "processed_sentences" in results.keys(), "Sentences must be present"

        # get tfidf features
        tfidf = TfidfVectorizer()
        tfidf_vectors = tfidf.fit_transform(results["processed_sentences"])

        # return task results
        task_results = {
            "tfidf_vectors": tfidf_vectors
        }
        return task_results


class TrainTask(Task):
    def __init__(self, id_: int, pre_task_ids: List[int], post_task_ids: List[int]):
        super().__init__(id_, pre_task_ids, post_task_ids)

    def run(self, results: Dict[str, Any]):
        assert "tfidf_vectors" in results.keys(), "TF-IDF Vectors must be present"
        assert "labels" in results.keys(), "Labels must be present"

        # fit a model here
        model = LogisticRegression()
        model.fit(results["tfidf_vectors"], results["labels"])

        # return task results
        task_results = {
            "model": model
        }

        return task_results


class EvaluateTask(Task):
    def __init__(self, id_: int, pre_task_ids: List[int], post_task_ids: List[int]):
        super().__init__(id_, pre_task_ids, post_task_ids)

    def run(self, results: Dict[str, Any]):
        assert "tfidf_vectors" in results.keys(), "TF-IDF Vectors must be present"
        assert "labels" in results.keys(), "Labels must be present"
        assert "model" in results.keys(), "Trained model must be present"

        # predict using model
        predictions = results["model"].predict(results["tfidf_vectors"])

        # evaluate
        report = classification_report(results["labels"], predictions)

        # return task results
        task_results = {
            "classification_report": report
        }

        return task_results


def main():
    tasks = [
        DatasetFetchTask(1, pre_task_ids=[], post_task_ids=[2, 3, 5]),
        PreProcessTask(2, pre_task_ids=[1], post_task_ids=[3]),
        FeaturizeTask(3, pre_task_ids=[2], post_task_ids=[4, 5]),
        TrainTask(4, pre_task_ids=[1, 3], post_task_ids=[5]),
        EvaluateTask(5, pre_task_ids=[1, 3, 4], post_task_ids=[])
    ]

    with Swarm(n_bees=3, refresh_every=5) as swarm:
        results = swarm.work(tasks)
    print(results[5]["classification_report"])


if __name__ == "__main__":
    main()
