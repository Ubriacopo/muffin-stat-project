from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import keras

from models.simple_cnn.one_conv_nn import OneConvNNModelFamily
from models.structure.base_model_family import BaseModelFamily, Channels, ConvLayerStructure, PoolLayerStructure


# So: Why use this and not generic?
#       1 - It is easier to tailor the search space.
#       2 - When tuning we are only considering a fixed number of layers forcefully which is a valid approach
#       3 - The make layers functions is easier to manipulate
#       4 - To add one "Layer" we can simply extend the TwoConvNet and add the needed field so the rewrite is only
#           on the make layers function. If we define the building procedure for the for loop in a function
#           this also gets easier as we can recall it without rewriting the loop
#       5 -  It is the result of extending previous models structure without loosing them. I could add a dense layer
#               in between if I wanted while with generic I still need to override the class.
class TwoConvNetFamily(OneConvNNModelFamily):

    def __init__(self, data_format: Channels = Channels.channels_last):
        """
        Simple CNN model. CNN that is simply structured by a sequence of convolutional layers followed by
        max pooling layers. At the end we have a single Dense activation hidden layer.
        """
        super().__init__(data_format)

        self.conv_structure_1: tuple[ConvLayerStructure, PoolLayerStructure | None] = (
            (ConvLayerStructure((16, 16), 2), PoolLayerStructure((2, 2), 2))
        )

    def make_layers(self, input_shape: (int, int, int)):

        x = None
        in_format = self.data_format.value

        input_layer = keras.Input(shape=input_shape, name=self.name)
        for conv, pool in [self.conv_structure_0, self.conv_structure_1]:
            x = keras.layers.Conv2D(filters=conv.filters, kernel_size=conv.kernel_size,
                                    data_format=in_format, activation="relu")(input_layer if x is None else x)

            if pool is not None:
                x = keras.layers.MaxPooling2D(pool_size=pool.pool_size, strides=pool.stride, data_format=in_format)(x)

        x = keras.layers.Flatten(data_format=self.data_format.value)(x)
        x = keras.layers.Dense(units=self.dense_layer_0_units, activation='relu')(x)

        output_layer = keras.layers.Dense(1, activation='sigmoid')(x)
        return input_layer, output_layer
