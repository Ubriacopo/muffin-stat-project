from __future__ import annotations

import enum
from abc import abstractmethod
from typing import Final

import keras


class Channels(enum.Enum):
    channels_last: str = 'channels_last'
    channels_first: str = 'channels_first'


class BaseModelWrapper:
    """
    A learning algorithm with one or more hyperparameters is not really an algorithm, but rather
    a family of algorithms, one for each possible assignment of values to the hyperparameters.
        ~ Hyperparameter tuning and risk estimates, Nicolò Cesa-Bianchi.

    The idea is that in the base model family we have ways to customize the learning algorithm
    to be an instance of the many possible of its family.
    """

    def __init__(self, data_format: Channels = Channels.channels_first):
        """
        The BaseModelWrapper (or Family as name is to be decided yet) wraps a family of learning algorithms
        in which we can vary the hyperparameters in use which might be relative to NN structure or learning process.

        :param data_format: Enum matching the data on the channels first or last based on the input data shape
        """
        self.data_format: Final[Channels] = data_format

    @abstractmethod
    def make_layers(self, input_shape: (int, int, int)) -> tuple[keras.Layer, keras.Layer]:
        """
        It defines the main structure of the model.
        :param input_shape:
        :return:
        """
        pass

    def make_model(self, input_shape: (int, int, int)) -> keras.Model:
        input_layer, output_layer = self.make_layers(input_shape=input_shape)
        return keras.Model(inputs=input_layer, outputs=output_layer)
