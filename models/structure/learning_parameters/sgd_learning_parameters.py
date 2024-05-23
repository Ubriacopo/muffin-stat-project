from __future__ import annotations

import keras
import keras_tuner

from models.structure.learning_parameters.learning_parameters import LearningParameters
from models.structure.learning_parameters.tunable_learning_parameters import \
    TunableLearningParameters


class SgdLearningParameters(LearningParameters):

    def __init__(self, learning_rate: float, loss: str | keras.losses.Loss = "binary_crossentropy",
                 metrics: list[str | keras.metrics.Metric] = None, momentum: float = 0.9, nesterov: bool = True):
        """

        :param learning_rate:
        :param loss:
        :param metrics:
        :param momentum:
        :param nesterov:
        """
        super().__init__(learning_rate, loss, metrics)
        self.momentum = momentum
        self.nesterov = nesterov

    def make_optimizer_instance(self) -> keras.optimizers.Optimizer:
        return keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=self.momentum, nesterov=self.nesterov)


class SgdLearningParametersTunable(SgdLearningParameters, TunableLearningParameters):
    def load_parameters(self, hp: keras_tuner.HyperParameters):
        self.learning_rate = hp.Float(name="lr", min_value=1e-5, max_value=1e-3, sampling='log', step=2)
        self.momentum = hp.Float(name="momentum", min_value=0.5, max_value=1, step=0.05)
