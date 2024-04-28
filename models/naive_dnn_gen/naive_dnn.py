from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Final

import keras
import keras_tuner

from models.structure.model_family import ModelFamily, TunableModelFamily


@dataclass
class HiddenLayerStructure:
    units: int
    following_dropout: float | None


class NaiveDNNModelFamily(ModelFamily):
    def __init__(self):
        """
        Naive DNN model.
        """
        super().__init__("naive_dnn", "binary_crossentropy")
        self.metrics: Final[list[str]] = ["accuracy"]

        self.hidden_layers: list[HiddenLayerStructure] = [
            HiddenLayerStructure(units=2048, following_dropout=None),
            HiddenLayerStructure(units=1024, following_dropout=0.3),
            HiddenLayerStructure(units=720, following_dropout=None),
        ]

    def make_layers(self, input_shape: (int, int, int)) -> tuple[keras.Layer, keras.Layer]:
        input_layer = keras.Input(shape=input_shape, batch_size=16, name=self.name)
        x = keras.layers.Flatten(data_format="channels_first")(input_layer)

        for layer_information in self.hidden_layers:
            x = keras.layers.Dense(units=layer_information.units, activation="relu")(x)
            x = (keras.layers.Dropout(rate=layer_information.following_dropout)
                 (x)) if layer_information.following_dropout is not None else x

        output_layer = keras.layers.Dense(units=1, activation='sigmoid')(x)
        return input_layer, output_layer

    def compile_model(self, model: keras.Model, optimizer: str | keras.optimizers.Optimizer):
        model.compile(optimizer=optimizer, metrics=self.metrics, loss=self.loss)


class NaiveDNNTunableModelFamily(NaiveDNNModelFamily, TunableModelFamily):
    def load_parameters(self, hp: keras_tuner.HyperParameters):
        self.hidden_layers = []  # Reset the default configuration.

        number_layers = hp.Int(name="layers", min_value=1, max_value=4, default=2)

        for i in range(number_layers):
            self.hidden_layers.append(HiddenLayerStructure(
                hp.Int(name=f"units_{i}", min_value=32, max_value=2048 - 512 * i, step=2, default=256, sampling='log'),
                0.50 if i + 1 < number_layers and hp.Boolean(name=f"dropout_{i}", default=False) else None
            ))
