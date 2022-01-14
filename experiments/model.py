import os
import torch
from typing import Union, List

from sentence_transformers import SentenceTransformer, util, InputExample, losses, CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sklearn.base import BaseEstimator, ClassifierMixin
from torch.utils.data import DataLoader

from experiments.utils import CROSS_ENCODERS


def is_cross_encoder(name):
    for cross_en in CROSS_ENCODERS:
        if cross_en in name:
            return True
    return False


class Model(BaseEstimator, ClassifierMixin):
    """
    Template intent recognizer compatible with scikit-learn.
    """

    T = Union[List[str], List[torch.Tensor]]

    def __init__(
        self, name, pretrained_model="", retrain=True, x_dev: List[List[str]] = None, y_dev: List[float] = None
    ):
        self._pretrained_model_path = pretrained_model
        self.model_file = pretrained_model if os.path.exists(self._pretrained_model_path) and not retrain else name

        if is_cross_encoder(self.model_file):
            self.model = CrossEncoder(self.model_file)
        else:
            self.model = SentenceTransformer(self.model_file)

        self.x_dev = x_dev or []
        self.y_dev = y_dev or []
        self.evaluator = None

        if dev := [InputExample(texts=sentences, label=score) for sentences, score in zip(self.x_dev, self.y_dev)]:
            if isinstance(self.model, CrossEncoder):
                self.evaluator = CECorrelationEvaluator.from_input_examples(dev, name="dev")
            else:
                self.evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev, name="dev")

    def fit(self, X: List[List[str]], y: List[float], retrain=True, epochs=5):
        """
        Fit the recognizer to the data.
        :param X: template sentences or vectors
        :param y: correct intents of X as indices
        """

        if os.path.exists(self._pretrained_model_path) and not retrain or not X:
            return self

        train_examples: List[InputExample] = []

        for sentences, score in zip(X, y):
            train_examples.append(InputExample(texts=sentences, label=score))

        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=8)
        train_loss = losses.CosineSimilarityLoss(self.model)
        if isinstance(self.model, CrossEncoder):
            train_objectives = train_dataloader
        else:
            train_objectives = [(train_dataloader, train_loss)]

        self.model.fit(
            train_objectives,
            epochs=epochs,
            warmup_steps=50,
            evaluator=self.evaluator,
            evaluation_steps=10,
            output_path=self._pretrained_model_path,
        )

        if is_cross_encoder(self.model_file):
            self.model = CrossEncoder(self._pretrained_model_path)
        else:
            self.model = SentenceTransformer(self._pretrained_model_path)
        return self

    def predict(self, X: List[List[str]]):
        return NotImplementedError

    def predict_proba(self, X: List[List[str]]):
        """
        Calculate STS for pairs of sentences
        :param X: list of sentence pairs
        :return: list of similarity score for each pair in X
        """

        if isinstance(self.model, CrossEncoder):
            return self.model.predict(X)

        vectors1 = self.model.encode([x[0] for x in X], convert_to_tensor=True)
        vectors2 = self.model.encode([x[1] for x in X], convert_to_tensor=True)
        return [util.pytorch_cos_sim(vec1, vec2).squeeze().tolist() for vec1, vec2 in zip(vectors1, vectors2)]
