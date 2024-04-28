from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import keras

from models.structure.base_model_family import BaseModelFamily, Channels


@dataclass
class ConvLayerStructure:
    kernel_size: tuple[int, int]
    filters: int


@dataclass
class PoolLayerStructure:
    pool_size: tuple[int, int]
    stride: int


# So: Why use this and not generic?
#       1 - It is easier to tailor the search space.
#       2 - When tuning we are only considering a fixed number of layers forcefully which is a valid approach
#       3 - The make layers functions is easier to manipulate
#       4 - To add one "Layer" we can simply extend the TwoConvNet and add the needed field so the rewrite is only
#           on the make layers function. If we define the building procedure for the for loop in a function
#           this also gets easier as we can recall it without rewriting the loop
#       5 -  It is the result of extending previous models structure without loosing them. I could add a dense layer
#               in between if I wanted while with generic I still need to override the class.
class TwoConvNetFamily(BaseModelFamily):

    def __init__(self, data_format: Channels = Channels.channels_last):
        """
        Simple CNN model. CNN that is simply structured by a sequence of convolutional layers followed by
        max pooling layers. At the end we have a single Dense activation hidden layer.
        """
        super().__init__("TwoConvNet", "binary_crossentropy")
        self.metrics: Final[list[str]] = ["accuracy"]

        self.conv_structure_0: tuple[ConvLayerStructure, PoolLayerStructure | None] = (
            (ConvLayerStructure((16, 16), 3), PoolLayerStructure((2, 2), 2))
        )

        self.conv_structure_1: tuple[ConvLayerStructure, PoolLayerStructure | None] = (
            (ConvLayerStructure((16, 16), 2), PoolLayerStructure((2, 2), 2))
        )

        self.dense_layer_units: int = 64

        # If the channels are first or not.
        self.data_format = data_format

    def make_layers(self, input_shape: (int, int, int)):

        x = None
        in_format = self.data_format.value

        input_layer = keras.Input(shape=input_shape, name=self.name)
        for _, structure in enumerate([self.conv_structure_0, self.conv_structure_1]):
            conv, pool = structure # Unpack the values of our structure.

            x = keras.layers.Conv2D(filters=conv.filters, kernel_size=conv.kernel_size,
                                    data_format=in_format, activation="relu")(input_layer if x is None else x)

            if pool is not None:
                x = keras.layers.MaxPooling2D(pool_size=pool.pool_size, strides=pool.stride, data_format=in_format)(x)

        x = keras.layers.Flatten(data_format=self.data_format.value)(x)
        x = keras.layers.Dense(units=self.dense_layer_units, activation='relu')(x)
        output_layer = keras.layers.Dense(1, activation='sigmoid')(x)

        return input_layer, output_layer

    def compile_model(self, model: keras.Model, optimizer: str | keras.optimizers.Optimizer):
        model.compile(optimizer=optimizer, metrics=self.metrics, loss=self.loss)
