import re
import string
from typing import Dict, Tuple, List

import numpy as np
from datasets import load_dataset
from flair.data import Sentence
from flair.embeddings import WordEmbeddings, DocumentPoolEmbeddings
from rich import print
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from fluidml import Task, Flow, TaskSpec
from fluidml.common.logging import configure_logging
from fluidml.storage import Sweep
from fluidml.visualization import visualize_graph_interactive


class DatasetFetchTask(Task):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _get_split(dataset, split):
        if split == "test":
            return dataset[split]
        elif split in ["train", "val"]:
            splitted = list(dataset["train"])
            split_index = int(0.7 * len(splitted))
            return splitted[:split_index] if split == "train" else splitted[split_index:]

    @staticmethod
    def _get_sentences_and_labels(dataset) -> Tuple[List[str], List[str]]:
        sentences = []
        labels = []
        for item in dataset:
            sentences.append(item["text"])
            labels.append(item["coarse_label"])
        return sentences, labels

    def run(self):
        dataset = load_dataset("trec")
        splits = ["train", "val", "test"]
        dataset_splits = {}
        for split in splits:
            dataset_split = DatasetFetchTask._get_split(dataset, split)
            sentences, labels = DatasetFetchTask._get_sentences_and_labels(dataset_split)
            split_results = {"sentences": sentences, "labels": labels}
            dataset_splits[split] = split_results
        self.save(dataset_splits, "raw_dataset")


class PreProcessTask(Task):
    def __init__(self, pre_processing_steps: List[str]):
        super().__init__()
        self._pre_processing_steps = pre_processing_steps

    def _pre_process(self, text: Dict) -> str:
        pre_processed_text = text
        for step in self._pre_processing_steps:
            if step == "lower_case":
                pre_processed_text = pre_processed_text.lower()
            if step == "remove_punct":
                pre_processed_text = pre_processed_text.translate(str.maketrans("", "", string.punctuation))
            if step == "remove_digits":
                pre_processed_text = re.sub(r"\d+", "<num>", pre_processed_text)
        return pre_processed_text

    def run(self, raw_dataset: Dict):
        pre_processed_splits = {}
        for split in ["train", "val", "test"]:
            pre_processed_sentences = [self._pre_process(sentence) for sentence in raw_dataset[split]["sentences"]]
            pre_processed_splits[split] = {"sentences": pre_processed_sentences}
        self.save(pre_processed_splits, "pre_processed_dataset")


class TFIDFFeaturizeTask(Task):
    def __init__(self, min_df: int, max_features: int):
        super().__init__()
        self._min_df = min_df
        self._max_features = max_features

    def run(self, pre_processed_dataset: Dict):
        tfidf_model = TfidfVectorizer(min_df=self._min_df, max_features=self._max_features)
        tfidf_model.fit(pre_processed_dataset["train"]["sentences"])
        featurized_splits = {}
        for split in ["train", "val", "test"]:
            tfidf_vectors = tfidf_model.transform(pre_processed_dataset[split]["sentences"]).toarray()
            featurized_splits[split] = {"vectors": tfidf_vectors}
        self.save(featurized_splits, "tfidf_featurized_dataset")


class GloveFeaturizeTask(Task):
    def __init__(self):
        super().__init__()

    def run(self, pre_processed_dataset: Dict):
        featurized_splits = {}
        for split in ["train", "val", "test"]:
            sentences = [Sentence(sent) for sent in pre_processed_dataset[split]["sentences"]]
            embedder = DocumentPoolEmbeddings([WordEmbeddings("glove")])
            embedder.embed(sentences)
            glove_vectors = [sent.embedding.cpu().numpy() for sent in sentences]
            glove_vectors = np.array(glove_vectors).reshape(len(glove_vectors), -1)
            featurized_splits[split] = {"vectors": glove_vectors}
        self.save(featurized_splits, "glove_featurized_dataset")


class TrainTask(Task):
    def __init__(self, max_iter: int, balanced: str):
        super().__init__()
        self._max_iter = max_iter
        self._class_weight = "balanced" if balanced else None

    def run(self, raw_dataset: Dict, tfidf_featurized_dataset: Dict, glove_featurized_dataset: Dict):
        model = LogisticRegression(max_iter=self._max_iter, class_weight=self._class_weight)
        stacked_vectors = np.hstack(
            (tfidf_featurized_dataset["train"]["vectors"], glove_featurized_dataset["train"]["vectors"])
        )
        model.fit(stacked_vectors, raw_dataset["train"]["labels"])
        self.save(model, "trained_model")


class EvaluateTask(Task):
    def __init__(self):
        super().__init__()

    def run(
        self,
        raw_dataset: Dict,
        tfidf_featurized_dataset: Dict,
        glove_featurized_dataset: Dict,
        trained_model: LogisticRegression,
    ):
        evaluation_results = {}
        for split in ["train", "val", "test"]:
            stacked_vectors = np.hstack(
                (tfidf_featurized_dataset[split]["vectors"], glove_featurized_dataset[split]["vectors"])
            )
            predictions = trained_model.predict(stacked_vectors)
            report = classification_report(raw_dataset[split]["labels"], predictions, output_dict=True)
            evaluation_results[split] = {"classification_report": report}
        self.save(evaluation_results, "evaluation_results")


class ModelSelectionTask(Task):
    def __init__(self):
        super().__init__()

    def run(self, evaluation_results: List[Sweep]):
        sorted_results = sorted(
            evaluation_results,
            key=lambda model_result: model_result.value["val"]["classification_report"]["macro avg"]["f1-score"],
            reverse=True,
        )
        self.save(sorted_results[0].config, "best_config")
        self.save(sorted_results[0].value, "best_performance")


def main():
    configure_logging(level="INFO")

    # create all task specs
    dataset_fetch_task = TaskSpec(task=DatasetFetchTask)
    pre_process_task = TaskSpec(task=PreProcessTask, config={"pre_processing_steps": ["lower_case", "remove_punct"]})
    featurize_task_1 = TaskSpec(task=GloveFeaturizeTask)
    featurize_task_2 = TaskSpec(
        task=TFIDFFeaturizeTask, config={"min_df": 5, "max_features": [1000, 2000]}, expand="product"
    )
    train_task = TaskSpec(task=TrainTask, config={"max_iter": [50, 100], "balanced": [True, False]}, expand="product")
    evaluate_task = TaskSpec(task=EvaluateTask)
    model_selection_task = TaskSpec(task=ModelSelectionTask, reduce=True)

    # dependencies between tasks
    pre_process_task.requires(dataset_fetch_task)
    featurize_task_1.requires(pre_process_task)
    featurize_task_2.requires(pre_process_task)
    train_task.requires([dataset_fetch_task, featurize_task_1, featurize_task_2])
    evaluate_task.requires([dataset_fetch_task, featurize_task_1, featurize_task_2, train_task])
    model_selection_task.requires(evaluate_task)

    # all tasks
    tasks = [
        dataset_fetch_task,
        pre_process_task,
        featurize_task_1,
        featurize_task_2,
        train_task,
        evaluate_task,
        model_selection_task,
    ]

    # create flow
    flow = Flow(tasks=tasks)

    # visualize graphs
    visualize_graph_interactive(flow.task_spec_graph)
    visualize_graph_interactive(flow.task_graph)

    # run the pipeline
    results = flow.run(project_name="sklearn_text_classification_example")

    print(results["ModelSelectionTask"]["results"]["best_config"])
    print(results["ModelSelectionTask"]["results"]["best_performance"])


if __name__ == "__main__":
    main()
