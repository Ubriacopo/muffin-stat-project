from __future__ import annotations

from typing import Final

import keras
import keras_tuner

from models.structure.base_model_family import BaseModelFamily, Channels, HiddenLayerStructure, ConvLayerStructure, \
    PoolLayerStructure
from models.structure.tunable_model_family_hypermodel import TunableModelFamily


class ConvNetFamily(BaseModelFamily):

    def __init__(self):
        """
        Simple CNN model. CNN that is simply structured by a sequence of convolutional layers followed by
        max pooling layers. At the end we have a single Dense activation hidden layer.
        """
        super().__init__("TwoConvNet", "binary_crossentropy")
        self.convolution_layers: list[tuple[ConvLayerStructure, PoolLayerStructure | None]] = [
            (ConvLayerStructure((16, 16), 3), PoolLayerStructure((2, 2), 2)),
            (ConvLayerStructure((16, 16), 2), PoolLayerStructure((2, 2), 2)),
        ]

        self.dense_layers: list[HiddenLayerStructure] = [
            HiddenLayerStructure(64, None),
        ]

    def make_layers(self, input_shape: (int, int, int)) -> tuple[keras.Layer, keras.Layer]:
        x = None

        # Structure ins ((CONV -> MAX_POOL) x i) -> ((DENSE -> DROPOUT?) x j) -> DENSE
        input_layer = keras.Input(shape=input_shape, name=self.name)
        for conv, pool in self.convolution_layers:
            # Convolution layer
            x = (keras.layers.Conv2D(filters=conv.filters, kernel_size=conv.kernel_size, padding='same',
                                     data_format=self.data_format.value, activation="relu")
                 (x if x is not None else input_layer))

            # Followed by a Pooling layer
            if pool is not None:
                x = keras.layers.MaxPool2D(pool_size=pool.pool_size, data_format=self.data_format.value)(x)

        x = keras.layers.Flatten(data_format=self.data_format.value)(x)

        # Dense Network
        for dense in self.dense_layers:
            x = keras.layers.Dense(units=dense.units, activation="relu")(x)

            if dense.following_dropout is not None:
                x = keras.layers.Dropout(rate=dense.following_dropout)(x)

        output_layer = keras.layers.Dense(1, activation='sigmoid')(x)
        return input_layer, output_layer


class TunableConvNetFamily(ConvNetFamily, TunableModelFamily):
    def load_parameters(self, hp: keras_tuner.HyperParameters):
        self.convolution_layers = []
        self.dense_layers = []

        for i in range(hp.Int("convolution_layers", min_value=1, max_value=2)):
            filters = hp.Int(f"filters_{i}", min_value=16, max_value=256, step=2, sampling='log')
            kernel_size = hp.Choice(f"kernel_{i}", values=[3, 5], default=3)

            self.convolution_layers.append((ConvLayerStructure(filters=filters, kernel_size=(kernel_size, kernel_size)),
                                            PoolLayerStructure(pool_size=(2, 2), stride=2)))

        for i in range(hp.Int("hidden_layers", min_value=1, max_value=3)):
            units = hp.Int(name=f"units_{i}", min_value=32, max_value=128, step=2, sampling='log')
            follow_dropout = 0.50 if hp.Boolean(name=f"dropout_{i}", default=False) else None
            self.dense_layers.append(HiddenLayerStructure(units, follow_dropout))