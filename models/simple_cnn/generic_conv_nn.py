from __future__ import annotations

from typing import Final

import keras
import keras_tuner

from models.simple_cnn.two_conv_nn import ConvLayerStructure, PoolLayerStructure
from models.structure.base_model_family import BaseModelFamily, Channels
from models.structure.default_augmented_model_family import InvertedChannelsAugmentedBasicModelFamily
from models.structure.tunable_model_family_hypermodel import TunableModelFamily

# I feel like this is not flexible enough
class GenericConvNetFamily(BaseModelFamily):

    def __init__(self, data_format: Channels = Channels.channels_last):
        """
        Simple CNN model. CNN that is simply structured by a sequence of convolutional layers followed by
        max pooling layers. At the end we have a single Dense activation hidden layer.
        """
        super().__init__("TwoConvNet", "binary_crossentropy")
        self.metrics: Final[list[str]] = ["accuracy"]

        self.network_structure: list[tuple[ConvLayerStructure, PoolLayerStructure | None]] = [
            (ConvLayerStructure((16, 16), 3), PoolLayerStructure((2, 2), 2)),
            (ConvLayerStructure((16, 16), 2), PoolLayerStructure((2, 2), 2)),
        ]

        self.dense_layer_units: int = 64

        # If the channels are first or not.
        self.data_format = data_format # todo give to root

    def make_layers(self, input_shape: (int, int, int)):
        x = None
        input_layer = keras.Input(shape=input_shape, name=self.name)

        for conv, pool in self.network_structure:

            x = keras.layers.Conv2D(
                filters=conv.filters, kernel_size=conv.kernel_size,
                data_format=self.data_format.value, activation="relu"
            )(x if x is not None else input_layer)

            if pool is not None:
                x = keras.layers.MaxPool2D(pool_size=pool.pool_size, data_format=self.data_format.value)(x)

        x = keras.layers.Flatten(data_format=self.data_format.value)(x)
        x = keras.layers.Dense(units=self.dense_layer_units, activation='relu')(x)
        output_layer = keras.layers.Dense(1, activation='sigmoid')(x)

        return input_layer, output_layer

    def compile_model(self, model: keras.Model, optimizer: str | keras.optimizers.Optimizer):
        model.compile(optimizer=optimizer, metrics=self.metrics, loss=self.loss)


class GenericConvTunableNetFamily(GenericConvNetFamily, TunableModelFamily):
    def load_parameters(self, hyperparameters: keras_tuner.HyperParameters):
        self.network_structure = [
            (
                ConvLayerStructure(
                    (
                        # We use square filters
                        hyperparameters.Int(name=f"filters_{i}", min_value=16, max_value=128, step=16),
                        hyperparameters.Int(name=f"filters_{i}", min_value=16, max_value=128, step=16)
                    ),
                    hyperparameters.Int(name=f"kernel_{i}", min_value=2, max_value=4)
                ),
                PoolLayerStructure(
                    (
                        # We use square pools
                        hyperparameters.Int(name=f"pool_size_{i}", min_value=16, max_value=128, step=16),
                        hyperparameters.Int(name=f"pool_size_{i}", min_value=16, max_value=128, step=16)
                    ),
                    hyperparameters.Int(name=f"stride_{i}", min_value=2, max_value=3))
            )
            for i in range(hyperparameters.Int(name="conv_layers", min_value=2, max_value=5))
        ]


class GenericConvTunableNetAugModelFamily(GenericConvTunableNetFamily, InvertedChannelsAugmentedBasicModelFamily):
    def __init__(self):
        super().__init__("TwoConvNet", "binary_crossentropy")
        # todo move higher in hierachy so we dont have to manually do it
        self.data_format = Channels.channels_last
