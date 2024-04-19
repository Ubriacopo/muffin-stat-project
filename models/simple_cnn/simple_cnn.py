from __future__ import annotations

from typing import Final

import keras
import keras_tuner

from models.tunable import TunableModelFamily


# Refactor like this https://keras.io/guides/keras_tuner/getting_started/#keep-keras-code-separate
class SimpleCnnModelFamily(TunableModelFamily):

    def __init__(self):
        super().__init__("SimpleCNN", "binary_crossentropy")
        self.metrics: Final[list[str]] = ["accuracy"]

        self.conv_layers: list[tuple[int, int, int]] = []
        self.hidden_units: int = 64

    def make_layers(self, input_shape: (int, int, int)):
        x = None
        input_layer = keras.Input(shape=input_shape, name=self.name)
        for kernel_size, filters in self.conv_layers:
            x = self.__make_conv_layer(kernel_size, filters, input_layer if x is None else x)

        x = keras.layers.Flatten(data_format="channels_first")(x)
        x = keras.layers.Dense(units=self.hidden_units, activation='relu')(x)
        output_layer = keras.layers.Dense(1, activation='sigmoid')(x)
        return input_layer, output_layer

    def compile_model(self, model: keras.Model, optimizer: str | keras.optimizers.Optimizer, **kwargs):
        model.compile(optimizer=optimizer, metrics=self.metrics, loss=self.loss)

    def load_parameters(self, hyperparameters):
        # todo: vedi se effettivamente funziona. sembra (non ho certezze) che facendo così se
        #           un valore di un hp è già definito allora la chiamata non fa niente.
        self.conv_layers = [[
            hyperparameters.Int(name=f"kernel_{i}", min_value=2, max_value=3),
            hyperparameters.Int(name=f"filters_{i}", min_value=16, max_value=128, step=16)
        ] for i in range(hyperparameters.Int(name="conv_layers", min_value=2, max_value=5))]

        self.hidden_units = hyperparameters.Int(name="hidden_units", min_value=32, max_value=128, default=64, step=16)

    def generate_search_parameters(self, hyperparameters: keras_tuner.HyperParameters | None = None):
        if hyperparameters is None:
            hyperparameters = keras_tuner.HyperParameters()

        hyperparameters.Int(name="conv_layers", min_value=1, max_value=3)
        hyperparameters.Int(name="hidden_units", min_value=16, max_value=128)

    def make_model(self, input_shape: (int, int, int)) -> keras.Model:
        # todo add augmentation optional now
        input_layer, output_layer = self.make_layers(input_shape=input_shape)
        return keras.Model(inputs=input_layer, outputs=output_layer)

    def __make_conv_layer(self, kernel_size, filters, previous_layer: keras.Layer) -> keras.Layer:
        x = keras.layers.Conv2D(filters=filters, kernel_size=(kernel_size, kernel_size),
                                padding='same', activation='relu', data_format='channels_first')(previous_layer)
        return keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, data_format='channels_first')(x)
