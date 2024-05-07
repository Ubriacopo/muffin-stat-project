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

    def load_learning_parameters(self, hp: keras_tuner.HyperParameters):
        self.optimizer = keras.optimizers.SGD(
            learning_rate=hp.Float(name="lr", min_value=1e-5, max_value=1e-3, sampling='log', step=2),
            momentum=hp.Float(name="momentum", min_value=0.5, max_value=1, sampling='reverse_log', step=2)
        )


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

        num_params = model.count_params()

        if num_params > 1870169702:  # Real estimate is 2077966336, but we take that -10% to be sure
            # When this error is raised, it skips the retries as memory not sufficient
            raise keras_tuner.errors.FailedTrialError(
                f"Model too large! It contains {num_params} params."
            )

        self.model_family.compile_model(model)
        return model
