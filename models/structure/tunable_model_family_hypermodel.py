import gc
from typing import Final

import keras
import keras_tuner
import torch

from models.structure.model_family import TunableModelFamily


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
        self.model_family.compile_model(model, hyperparameters.get("optimizer"))
        return model
