from typing import Dict, Any, Tuple, List

from datasets import load_dataset
from flair.data import Sentence
from flair.embeddings import WordEmbeddings, DocumentPoolEmbeddings
import numpy as np
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from rich import print

from fluidml.common import Task, Resource
from fluidml.swarm import Swarm
from fluidml.flow import Flow, TaskSpec


class DatasetFetchTask(Task):
    def __init__(self):
        super().__init__()

    def _get_split(self, dataset, split):
        if split == "test":
            return dataset[split]
        elif split in ["train", "val"]:
            splitted = list(dataset["train"])
            split_index = int(0.7 * len(splitted))
            return splitted[:split_index] if split == "train" else splitted[split_index:]

    def _get_sentences_and_labels(self, dataset) -> Tuple[List[str], List[str]]:
        sentences = []
        labels = []
        for item in dataset:
            sentences.append(item["text"])
            labels.append(item["label-coarse"])
        return sentences, labels

    def run(self, results: Dict[str, Any], resource: Resource):
        dataset = load_dataset("trec")
        splits = ["train", "val", "test"]
        task_results = {}
        for split in splits:
            dataset_split = self._get_split(dataset, split)
            sentences, labels = self._get_sentences_and_labels(dataset_split)
            split_results = {
                "sentences": sentences,
                "labels": labels
            }
            task_results[split] = split_results
        return task_results


class PreProcessTask(Task):
    def __init__(self, pre_processing_steps: List[str]):
        super().__init__()
        self._pre_processing_steps = pre_processing_steps

    def _pre_process(self, text: str) -> str:
        pre_processed_text = text
        for step in self._pre_processing_steps:
            if step == "lower_case":
                pre_processed_text = pre_processed_text.lower()
            if step == "remove_punct":
                pre_processed_text = pre_processed_text.translate(
                    str.maketrans('', '', string.punctuation))
            if step == "remove_digits":
                pre_processed_text = re.sub(
                    r"\d+", "<num>", pre_processed_text)
        return pre_processed_text

    def run(self, results: Dict[str, Any], resource: Resource):
        task_results = {}
        for split in ["train", "val", "test"]:
            pre_processed_sentences = [
                self._pre_process(sentence) for sentence in results["dataset"]["result"][split]["sentences"]]
            task_results[split] = {"sentences": pre_processed_sentences}
        return task_results


class TFIDFFeaturizeTask(Task):
    def __init__(self, min_df: int, max_features: int):
        super().__init__()
        self._min_df = min_df
        self._max_features = max_features

    def run(self, results: Dict[str, Any], resource: Resource):
        tfidf_model = TfidfVectorizer(
            min_df=self._min_df, max_features=self._max_features)
        tfidf_model.fit(results["pre_process"]["result"]["train"]["sentences"])
        task_results = {}
        for split in ["train", "val", "test"]:
            tfidf_vectors = tfidf_model.transform(
                results["pre_process"]["result"][split]["sentences"]).toarray()
            task_results[split] = {"vectors": tfidf_vectors}
        return task_results


class GloveFeaturizeTask(Task):
    def __init__(self):
        super().__init__()

    def run(self, results: Dict[str, Any], resource: Resource):
        task_results = {}
        for split in ["train", "val", "test"]:
            sentences = [Sentence(sent)
                         for sent in results["pre_process"]["result"][split]["sentences"]]
            embedder = DocumentPoolEmbeddings([WordEmbeddings("glove")])
            embedder.embed(sentences)
            glove_vectors = [sent.embedding.cpu().numpy()
                             for sent in sentences]
            glove_vectors = np.array(glove_vectors).reshape(
                len(glove_vectors), -1)
            task_results[split] = {"vectors": glove_vectors}
        return task_results


class TrainTask(Task):
    def __init__(self, max_iter: int, class_weight: str):
        super().__init__()
        self._max_iter = max_iter
        self._class_weight = class_weight

    def run(self, results: Dict[str, Any], resource: Resource):
        model = LogisticRegression(
            max_iter=self._max_iter, class_weight=self._class_weight)
        stacked_vectors = np.hstack((results["tfidf_featurize"]["result"]["train"]["vectors"],
                                     results["glove_featurize"]["result"]["train"]["vectors"]))
        model.fit(stacked_vectors,
                  results["dataset"]["result"]["train"]["labels"])
        task_results = {
            "model": model
        }
        return task_results


class EvaluateTask(Task):
    def __init__(self):
        super().__init__()

    def run(self, results: Dict[str, Any], resource: Resource):
        task_results = {}
        for split in ["train", "val", "test"]:
            stacked_vectors = np.hstack((results["tfidf_featurize"]["result"][split]["vectors"],
                                         results["glove_featurize"]["result"][split]["vectors"]))
            predictions = results["train"]["result"]["model"].predict(
                stacked_vectors)
            report = classification_report(
                results["dataset"]["result"][split]["labels"], predictions, output_dict=True)
            task_results[split] = {"classification_report": report}
        return task_results


def main():

    # create all task specs
    dataset_fetch_task = TaskSpec(task=DatasetFetchTask, name="dataset")
    pre_process_task = TaskSpec(task=PreProcessTask, name="pre_process", task_kwargs={
                                "pre_processing_steps": ["lower_case", "remove_punct"]})
    featurize_task_1 = TaskSpec(
        task=GloveFeaturizeTask, name="glove_featurize")
    featurize_task_2 = TaskSpec(
        task=TFIDFFeaturizeTask, name="tfidf_featurize", task_kwargs={"min_df": 5, "max_features": 1000})
    train_task = TaskSpec(task=TrainTask, name="train",
                          task_kwargs={"max_iter": 50, "class_weight": "balanced"})
    evaluate_task = TaskSpec(task=EvaluateTask, name="evaluate")

    # dependencies between tasks
    pre_process_task.requires([dataset_fetch_task])
    featurize_task_1.requires([pre_process_task])
    featurize_task_2.requires([pre_process_task])
    train_task.requires(
        [dataset_fetch_task, featurize_task_1, featurize_task_2])
    evaluate_task.requires(
        [dataset_fetch_task, featurize_task_1, featurize_task_2, train_task])

    # all tasks
    tasks = [dataset_fetch_task,
             pre_process_task,
             featurize_task_1, featurize_task_2,
             train_task,
             evaluate_task]

    with Swarm(n_dolphins=1,
               refresh_every=10,
               return_results=True) as swarm:
        flow = Flow(swarm=swarm)
        results = flow.run(tasks)
    print(results["evaluate"]["result"])


if __name__ == "__main__":
    main()
