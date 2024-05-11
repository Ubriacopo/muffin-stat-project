from abc import abstractmethod

import keras_tuner

from models.structure.base_model_wrapper import BaseModelWrapper


class TunableWrapperBase(BaseModelWrapper):
    @abstractmethod
    def load_parameters(self, hyperparameters: keras_tuner.HyperParameters):
        pass
