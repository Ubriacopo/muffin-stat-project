from __future__ import annotations

from typing import Final

import keras
import keras_tuner

from models.structure.model_family import ModelFamily, TunableModelFamily, \
    FinalModelFamilyInstance


class SimpleCnnModelFamily(ModelFamily):

    def __init__(self):
        """
        Simple CNN model. CNN that is simply structured by a sequence of convolutional layers followed by
        max pooling layers. At the end we have a single Dense activation hidden layer.
        """
        super().__init__("SimpleCNN", "binary_crossentropy")
        self.metrics: Final[list[str]] = ["accuracy"]
        # todo: Poco chiaro così. Un tipo apposito potrebbe essere più chiaro
        self.conv_layers: list[tuple[int, int]] = [(16, 3), (16, 2)]
        self.hidden_units: int = 64

    def make_layers(self, input_shape: (int, int, int)):
        x = None

        input_layer = keras.Input(shape=input_shape, name=self.name)
        for kernel_size, filters in self.conv_layers:
            x = self.make_conv_layer(kernel_size, filters, input_layer if x is None else x)

        x = keras.layers.Flatten(data_format="channels_first")(x)
        x = keras.layers.Dense(units=self.hidden_units, activation='relu')(x)
        output_layer = keras.layers.Dense(1, activation='sigmoid')(x)

        return input_layer, output_layer

    @staticmethod
    def make_conv_layer(kernel_size, filters, previous_layer: keras.Layer) -> keras.Layer:
        x = keras.layers.Conv2D(filters=filters, kernel_size=(kernel_size, kernel_size),
                                padding='same', activation='relu', data_format='channels_first')(previous_layer)
        return keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, data_format='channels_first')(x)

    def compile_model(self, model: keras.Model, optimizer: str | keras.optimizers.Optimizer):
        model.compile(optimizer=optimizer, metrics=self.metrics, loss=self.loss)


# Refactor like this https://keras.io/guides/keras_tuner/getting_started/#keep-keras-code-separate
class SimpleCnnTunableModelFamily(SimpleCnnModelFamily, TunableModelFamily):
    def load_parameters(self, hyperparameters):
        # todo: vedi se effettivamente funziona. sembra (non ho certezze) che facendo così se
        #           un valore di un hp è già definito allora la chiamata non fa niente.
        self.conv_layers = [[
            hyperparameters.Int(name=f"kernel_{i}", min_value=2, max_value=3),
            hyperparameters.Int(name=f"filters_{i}", min_value=16, max_value=128, step=16)
        ] for i in range(hyperparameters.Int(name="conv_layers", min_value=2, max_value=5))]

        self.hidden_units = hyperparameters.Int(name="hidden_units", min_value=32, max_value=128, default=64, step=16)


class FinalSimpleCnnModel(FinalModelFamilyInstance):
    """
    todo valuta se buttarlo via
    Best configuration parameters for the SimpleCnn are so fixed without using hyperparameters
    or explicitly calling them. This is to better showcase the structure of keras network.
    """

    def make_layers(self, input_shape: (int, int, int)):
        input_layer = keras.Input(shape=input_shape, name=self.name)

        x = SimpleCnnModelFamily.make_conv_layer(3, 16, input_layer)
        x = SimpleCnnModelFamily.make_conv_layer(2, 16, x)

        x = keras.layers.Flatten(data_format="channels_first")(x)
        x = keras.layers.Dense(units=32, activation='relu')(x)

        output_layer = keras.layers.Dense(1, activation='sigmoid')(x)
        return input_layer, output_layer

    def compile_model(self, model: keras.Model, optimizer: str | keras.optimizers.Optimizer, **kwargs):
        model.compile(optimizer=optimizer, metrics=['accuracy'], loss='binary_crossentropy')

    def get_model(self, input_shape: (int, int, int), default_compile: bool = True) -> keras.Model:
        model = self.make_model(input_shape)

        if default_compile:
            optimizer = keras.optimizers.Adam(lr=1e-4)
            self.compile_model(model, optimizer)

        return model
