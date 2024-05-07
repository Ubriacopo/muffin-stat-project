from __future__ import annotations

import enum
from abc import abstractmethod
from dataclasses import dataclass
from typing import Final

import keras


class Channels(enum.Enum):
    channels_last: str = 'channels_last'
    channels_first: str = 'channels_first'


class BaseModelFamily:
    """
    A learning algorithm with one or more hyperparameters is not really an algorithm, but rather
    a family of algorithms, one for each possible assignment of values to the hyperparameters.
        ~ Hyperparameter tuning and risk estimates, Nicolò Cesa-Bianchi.

    The idea is that in the base model family we have ways to customize the learning algorithm
    to be an instance of the many possible of its family.
    """

    def __init__(self, family_name: str, loss: str, metrics: list[str] = None,
                 data_format: Channels = Channels.channels_first):
        self.name: Final[str] = family_name
        self.loss: Final[str] = loss

        self.metrics: Final[list[str]] = metrics if metrics is not None else ["accuracy"]
        self.data_format: Final[Channels] = data_format

    @abstractmethod
    def make_layers(self, input_shape: (int, int, int)) -> tuple[keras.Layer, keras.Layer]:
        """
        It defines the main structure of the model.
        :param input_shape:
        :return:
        """
        pass

    def compile_model(self, model: keras.Model):
        model.compile(optimizer=keras.optimizers.SGD(learning_rate=1e-4, momentum=0.9, nesterov=True),
                      metrics=self.metrics, loss=self.loss)

    def make_model(self, input_shape: (int, int, int)) -> keras.Model:
        input_layer, output_layer = self.make_layers(input_shape=input_shape)
        return keras.Model(inputs=input_layer, outputs=output_layer)


@dataclass
class HiddenLayerStructure:
    units: int
    following_dropout: float | None


@dataclass
class ConvLayerStructure:
    kernel_size: tuple[int, int]
    filters: int


@dataclass
class PoolLayerStructure:
    pool_size: tuple[int, int]
    stride: int
