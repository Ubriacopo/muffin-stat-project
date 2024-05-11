from abc import abstractmethod

import keras_tuner

from models.structure.learning_parameters.learning_parameters import LearningParameters


class TunableLearningParameters(LearningParameters):
    @abstractmethod
    def load_parameters(self, hyperparameters: keras_tuner.HyperParameters):
        pass
