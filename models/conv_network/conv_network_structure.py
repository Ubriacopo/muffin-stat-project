from __future__ import annotations

import keras
import keras_tuner
from keras.src.layers import Conv2D, MaxPooling2D, Dropout, Dense, GlobalAveragePooling2D, Flatten

from models.structure.base_model_wrapper import BaseModelWrapper
from models.structure.layer_structure_data import ConvLayerStructure, PoolLayerStructure, HiddenLayerStructure, \
    DropoutLayerStructure
from models.structure.tunable_wrapper import TunableWrapperBase


# The class is not bound to any specific structure, but we will be using it as:
# Structure ins in:((CONV -> MAX_POOL -> DROPOUT?) x i) -> ((DENSE -> DROPOUT?) x j) -> DENSE:out
# The structure is frozen if someone wants to define the layers he has override the class
# or ignore the fact that we give a tuple and go all hacky on the code.
class ConvNetworkStructure(BaseModelWrapper):
    convolutional_layers: tuple[ConvLayerStructure | PoolLayerStructure | DropoutLayerStructure] = [
        ConvLayerStructure((3, 3), 16),
        PoolLayerStructure.default(),
        ConvLayerStructure((3, 3), 32),
        PoolLayerStructure.default(),
    ]
    dense_layers: tuple[HiddenLayerStructure | DropoutLayerStructure] = [
        HiddenLayerStructure(64)
    ]

    def make_layers(self, input_shape: (int, int, int)) -> tuple[keras.Layer, keras.Layer]:
        input_layer = keras.Input(shape=input_shape, name=self.__class__.__name__)
        x = input_layer
        chan = self.data_format.value

        for layer in self.convolutional_layers:
            if isinstance(layer, ConvLayerStructure):
                x = Conv2D(filters=layer.filters, kernel_size=layer.kernel_size,
                           padding="same", data_format=chan, activation='relu')(x)
            if isinstance(layer, PoolLayerStructure):
                x = MaxPooling2D(pool_size=layer.pool_size, data_format=chan)(x)
            if isinstance(layer, DropoutLayerStructure):
                x = Dropout(rate=layer.rate)(x)

        # https://stackabuse.com/dont-use-flatten-global-pooling-for-cnns-with-tensorflow-and-keras/ ??
        x = Flatten()(x)

        for layer in self.dense_layers:
            if isinstance(layer, HiddenLayerStructure):
                x = Dense(units=layer.units, activation="relu")(x)
            if isinstance(layer, DropoutLayerStructure):
                x = Dropout(rate=layer.rate)(x)

        output_layer = keras.layers.Dense(1, activation='sigmoid')(x)
        return input_layer, output_layer


class TunableConvNetworkStructure(ConvNetworkStructure, TunableWrapperBase):
    def load_parameters(self, hp: keras_tuner.HyperParameters):
        self.convolutional_layers: list = []
        self.dense_layers: list = []

        for i in range(hp.Int("convolution_layers", min_value=1, max_value=4)):
            filters = hp.Int(f"filters_{i}", min_value=16, max_value=256, step=2, sampling='log')
            kernel_size = hp.Choice(f"kernel_{i}", values=[3, 5], default=3)
            self.convolutional_layers.append(ConvLayerStructure((kernel_size, kernel_size), filters))
            self.convolutional_layers.append(PoolLayerStructure.default())

            # When learning the network we decide that regularization takes place later.
            if False and hp.Choice(f"conv_dropout_{i}", values=[True, False], default=False):
                self.convolutional_layers.append(DropoutLayerStructure(0.2))

        for i in range(hp.Int("hidden_layers", min_value=1, max_value=2)):
            units = hp.Int(name=f"units_{i}", min_value=32, max_value=256, step=2, sampling='log')
            self.dense_layers.append(HiddenLayerStructure(units))

            # When learning the network we decide that regularization takes place later.
            if False and hp.Boolean(name=f"dropout_{i}", default=False):
                self.dense_layers.append(DropoutLayerStructure(0.5))
