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
from fluidml.flow import Flow, GridTaskSpec, TaskSpec


def results_available(results, task_name, value) -> bool:
    return results.get(task_name, None) is not None and \
        results[task_name].get("result", None) is not None and  \
        results[task_name]["result"].get(value, None) is not None


class DatasetFetchTask(Task):
    def __init__(self, name: str, id_: int):
        super().__init__(name, id_)

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
    def __init__(self, name: str, id_: int, pre_processing_steps: List[str]):
        super().__init__(name, id_)
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
    def __init__(self, name: str, id_: int, min_df: int, max_features: int):
        super().__init__(name, id_)
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
    def __init__(self, name: str, id_: int):
        super().__init__(name, id_)

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
    def __init__(self, name: str, id_: int, max_iter: int, balanced: str):
        super().__init__(name, id_)
        self._max_iter = max_iter
        self._class_weight = "balanced" if balanced else None

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
    def __init__(self, name: str, id_: int):
        super().__init__(name, id_)

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


class ModelSelectionTask(Task):
    def __init__(self, name: str, id_: int):
        super().__init__(name, id_)

    def run(self, results: Dict[str, Any], resource: Resource):
        sorted_results = sorted(results["evaluate"], key=lambda model_result: model_result["result"]
                                ["val"]["classification_report"]["macro avg"]["f1-score"], reverse=True)
        task_results = {
            "best_config": sorted_results[0]["config"],
            "best_performance": sorted_results[0]["result"]
        }
        return task_results


def main():

    # create all task specs
    dataset_fetch_task = TaskSpec(task=DatasetFetchTask, name="dataset")
    pre_process_task = GridTaskSpec(task=PreProcessTask, name="pre_process", gs_config={
        "pre_processing_steps": ["lower_case", "remove_punct"]})
    featurize_task_1 = TaskSpec(
        task=GloveFeaturizeTask, name="glove_featurize")
    featurize_task_2 = GridTaskSpec(
        task=TFIDFFeaturizeTask, name="tfidf_featurize", gs_config={"min_df": 5, "max_features": [1000, 2000]})
    train_task = GridTaskSpec(task=TrainTask, name="train",
                              gs_config={"max_iter": [50, 100], "balanced": [True, False]})
    evaluate_task = TaskSpec(task=EvaluateTask, name="evaluate")
    model_selection_task = TaskSpec(
        task=ModelSelectionTask, name="model_select", reduce=True)

    # dependencies between tasks
    pre_process_task.requires([dataset_fetch_task])
    featurize_task_1.requires([pre_process_task])
    featurize_task_2.requires([pre_process_task])
    train_task.requires(
        [dataset_fetch_task, featurize_task_1, featurize_task_2])
    evaluate_task.requires(
        [dataset_fetch_task, featurize_task_1, featurize_task_2, train_task])
    model_selection_task.requires([evaluate_task])

    # all tasks
    tasks = [dataset_fetch_task,
             pre_process_task,
             featurize_task_1, featurize_task_2,
             train_task,
             evaluate_task,
             model_selection_task]

    with Swarm(n_dolphins=2,
               refresh_every=10,
               return_results=True) as swarm:
        flow = Flow(swarm=swarm)
        results = flow.run(tasks)
    print(results["model_select"]["result"]["best_config"])
    print(results["model_select"]["result"]["best_performance"])


if __name__ == "__main__":
    main()
