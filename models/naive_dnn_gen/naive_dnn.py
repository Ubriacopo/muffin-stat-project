from __future__ import annotations

from abc import ABC
from typing import Final

import keras
import keras_tuner

from models.structure.model_family import ModelFamily, TunableModelFamily


class NaiveDNNModelFamily(ModelFamily):
    def __init__(self):
        """
        Naive DNN model.
        """
        super().__init__("naive_dnn", "binary_crossentropy")
        self.metrics: Final[list[str]] = ["accuracy"]
        # todo: tipo piu preciso per hidden layers?
        self.hidden_layers: list[tuple[int, float | None]] = [
            (2048, None), (1024, None), (720, 0.3)
        ]

    def make_layers(self, input_shape: (int, int, int)) -> tuple[keras.Layer, keras.Layer]:
        input_layer = keras.Input(shape=input_shape, batch_size=16, name=self.name)
        x = keras.layers.Flatten(data_format="channels_first")(input_layer)
        for layer_units, dropout in self.hidden_layers:
            x = keras.layers.Dense(units=layer_units, activation="relu")(x)
            x = keras.layers.Dropout(rate=dropout)(x) if dropout is not None else x

        output_layer = keras.layers.Dense(units=1, activation='sigmoid')(x)
        return input_layer, output_layer

    def compile_model(self, model: keras.Model, optimizer: str | keras.optimizers.Optimizer):
        model.compile(optimizer=optimizer, metrics=self.metrics, loss=self.loss)


class NaiveDNNTunableModelFamily(NaiveDNNModelFamily, TunableModelFamily):
    def load_parameters(self, hyperparameters):
        self.hidden_layers = []  # Reset the default configuration.
        number_layers = hyperparameters.Int(name="layers", min_value=1, max_value=4, default=2)
        for i in range(number_layers):
            self.hidden_layers.append((
                hyperparameters.Int(
                    name=f"units_{i}", min_value=32, max_value=1024, step=2, default=256, sampling='log'
                ),
                0.30 if i + 1 < number_layers and hyperparameters.Boolean(name=f"dropout_{i}", default=False) else None
            ))
        print(self.hidden_layers)
