from __future__ import annotations

from abc import abstractmethod
from typing import Final

import keras


class LearningParameters:
    def __init__(self, learning_rate: float, loss: str | keras.losses.Loss = "binary_crossentropy",
                 metrics: list[str | keras.metrics.Metric] = None):
        """

        :param learning_rate: The learning rate selected for the optimizer.
        :param loss: A loss function which might be a string or a keras.losses.Loss instance. (The string
        references a loss function of keras that is inferred automatically)
        :param metrics: List of metrics to track in the model compile.
        """
        self.learning_rate = learning_rate

        self.loss: Final[str | keras.losses.Loss] = loss
        self.metrics: Final[list[str | keras.metrics.Metric]] = metrics if metrics is not None else ["accuracy"]

    def compile_model(self, model: keras.Model):
        model.compile(optimizer=self.make_optimizer_instance(), metrics=self.metrics, loss=self.loss)

    @abstractmethod
    def make_optimizer_instance(self) -> keras.optimizers.Optimizer:
        pass
