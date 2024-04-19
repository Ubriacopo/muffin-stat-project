from __future__ import annotations

import gc
from abc import abstractmethod
from typing import Final

import keras
import keras_tuner
import torch


class Tunable(keras_tuner.HyperModel):

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


class ModelFamily:

    def __init__(self, family_name: str, loss: str):
        self.name: Final[str] = family_name
        self.loss: Final[str] = loss

    @abstractmethod
    def make_model(self, input_shape: (int, int, int)) -> keras.Model:
        pass

    @abstractmethod
    def make_layers(self, input_shape: (int, int, int)):
        """
        It defines the main structure of the model.
        :param input_shape:
        :param kwargs:
        :return:
        """
        pass

    @abstractmethod
    def compile_model(self, model: keras.Model, optimizer: str | keras.optimizers.Optimizer, **kwargs):
        pass


class TunableModelFamily(ModelFamily):
    @abstractmethod
    def load_parameters(self, hyperparameters: keras_tuner.HyperParameters):
        pass

    @abstractmethod
    def generate_search_parameters(self, hyperparameters: keras_tuner.HyperParameters | None = None):
        pass
