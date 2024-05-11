from __future__ import annotations

import keras
import keras_tuner

from models.structure.base_model_family import BaseModelFamily, Channels, ConvLayerStructure, PoolLayerStructure
from models.structure.tunable_model_family_hypermodel import TunableModelFamily


class OneConvNNModelFamily(BaseModelFamily):
    conv_structure_0: tuple[ConvLayerStructure, PoolLayerStructure | None] = (
        (ConvLayerStructure((16, 16), 3), PoolLayerStructure((2, 2), 2))
    )

    dense_layer_0_units: int = 64

    def make_layers(self, input_shape: (int, int, int)):
        in_format = self.data_format.value

        input_layer = keras.Input(shape=input_shape, name=self.__class__.__name__)
        conv, pool = self.conv_structure_0

        x = keras.layers.Conv2D(filters=conv.filters, kernel_size=conv.kernel_size, data_format=in_format,
                                activation="relu", padding='same')(input_layer)

        x = keras.layers.MaxPooling2D(pool_size=pool.pool_size, strides=pool.stride, data_format=in_format)(x)
        x = keras.layers.Flatten(data_format=self.data_format.value)(x)
        x = keras.layers.Dense(units=self.dense_layer_0_units, activation='relu')(x)

        output_layer = keras.layers.Dense(1, activation='sigmoid')(x)
        return input_layer, output_layer


class OneConvNNTunableModelFamily(OneConvNNModelFamily, TunableModelFamily):
    def load_parameters(self, hp: keras_tuner.HyperParameters):
        filters = hp.Int("filters_0", min_value=16, max_value=256, step=2, sampling='log')
        kernel_size = hp.Choice("kernel_0", values=[3, 5])

        self.conv_structure_0 = (
            ConvLayerStructure(filters=filters, kernel_size=(kernel_size, kernel_size)),
            PoolLayerStructure(pool_size=(2, 2), stride=2)
        )

        # Learning parameters
