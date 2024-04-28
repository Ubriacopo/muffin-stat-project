from __future__ import annotations

import enum
from abc import abstractmethod
from dataclasses import dataclass
from typing import Final

import keras


class BaseModelFamily:
    """
    A learning algorithm with one or more hyperparameters is not really an algorithm, but rather
    a family of algorithms, one for each possible assignment of values to the hyperparameters.
        ~ Hyperparameter tuning and risk estimates, Nicolò Cesa-Bianchi.

    The idea is that in the base model family we have ways to customize the learning algorithm
    to be an instance of the many possible of its family.
    """

    def __init__(self, family_name: str, loss: str):
        self.name: Final[str] = family_name
        self.loss: Final[str] = loss

    @abstractmethod
    def make_layers(self, input_shape: (int, int, int)) -> tuple[keras.Layer, keras.Layer]:
        """
        It defines the main structure of the model.
        :param input_shape:
        :return:
        """
        pass

    @abstractmethod
    def compile_model(self, model: keras.Model, optimizer: str | keras.optimizers.Optimizer | None):
        pass

    def make_model(self, input_shape: (int, int, int)) -> keras.Model:
        input_layer, output_layer = self.make_layers(input_shape=input_shape)
        return keras.Model(inputs=input_layer, outputs=output_layer)


@dataclass
class HiddenLayerStructure:
    units: int
    following_dropout: float | None


class Channels(enum.Enum):
    channels_last: str = 'channels_last'
    channels_first: str = 'channels_first'
