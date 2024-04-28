from __future__ import annotations
import gc
from abc import abstractmethod
from typing import Final

import keras
import keras_tuner
import torch

from models.structure.base_model_family import BaseModelFamily


class TunableModelFamily(BaseModelFamily):
    @abstractmethod
    def load_parameters(self, hyperparameters: keras_tuner.HyperParameters):
        pass


class TunableModelFamilyHypermodel(keras_tuner.HyperModel):

    def __init__(self, input_shape: (int, int, int), model_family: TunableModelFamily):
        super().__init__()

        self.input_shape: Final[(int, int, int)] = input_shape
        self.model_family: Final[TunableModelFamily] = model_family

    def build(self, hyperparameters: keras_tuner.HyperParameters) -> keras.Model:
        # Release memory to avoid OOM during tuning.
        torch.cuda.empty_cache()
        gc.collect()

        self.model_family.load_parameters(hyperparameters)

        model = self.model_family.make_model(self.input_shape)
        self.model_family.compile_model(model, keras.optimizers.SGD(
            learning_rate=hyperparameters.Float(name="lr", min_value=1e-5, max_value=1e-3, sampling='log', step=2),
            momentum=hyperparameters.Float(name="momentum", min_value=0.5, max_value=1, sampling='reverse_log', step=2)
        ))
        return model