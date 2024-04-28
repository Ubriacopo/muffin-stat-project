from __future__ import annotations

from typing import Final

import keras
import keras_tuner

from models.structure.default_augmented_model_family import ChannelsLastFixAugmentedNaiveDNNModelFamily
from models.structure.base_model_family import BaseModelFamily, HiddenLayerStructure
from models.structure.tunable_model_family_hypermodel import TunableModelFamily


class TwoHiddenLayersDNNModelFamily(BaseModelFamily):
    def __init__(self):
        super().__init__("AugmentedDNN", "binary_crossentropy")

        self.hidden_layer_0 = HiddenLayerStructure(units=2048, following_dropout=0.5)
        self.hidden_layer_1 = HiddenLayerStructure(units=720, following_dropout=0.5)

        self.metrics: Final[list[str]] = ["accuracy"]

    def make_layers(self, input_shape: (int, int, int)) -> tuple[keras.Layer, keras.Layer]:
        input_layer = keras.layers.Input(shape=input_shape, name=self.name)
        x = keras.layers.Flatten()(input_layer)

        x = keras.layers.Dense(units=self.hidden_layer_0.units, activation="relu")(x)
        if self.hidden_layer_0.following_dropout is not None:
            x = keras.layers.Dropout(self.hidden_layer_0.following_dropout)(x)

        x = keras.layers.Dense(units=self.hidden_layer_1.units, activation="relu")(x)
        if self.hidden_layer_1.following_dropout is not None:
            x = keras.layers.Dropout(self.hidden_layer_1.following_dropout)(x)

        output_layer = keras.layers.Dense(units=1, activation='sigmoid')(x)
        return input_layer, output_layer

    def compile_model(self, model: keras.Model, optimizer: str | keras.optimizers.Optimizer | None):
        if optimizer is None:
            optimizer = keras.optimizers.SGD(learning_rate=1e-4, momentum=0.9, nesterov=True)

        model.compile(optimizer=optimizer, metrics=self.metrics, loss=self.loss)


class TwoHiddenLayersTunableAugmentedDNN(TwoHiddenLayersDNNModelFamily, TunableModelFamily):
    def load_parameters(self, hp: keras_tuner.HyperParameters):
        self.hidden_layer_0 = HiddenLayerStructure(
            hp.Int(name=f"units_0", min_value=1024, max_value=4096, step=256, default=1024),
            0.50 if hp.Boolean(name=f"dropout_0", default=False) else None
        )

        self.hidden_layer_1 = HiddenLayerStructure(
            hp.Int(name=f"units_1", min_value=128, max_value=1536, step=128, default=256),
            0.50 if hp.Boolean(name=f"dropout_1", default=False) else None
        )


class TwoHiddenLayersDNNAugModelFamily(TwoHiddenLayersTunableAugmentedDNN, ChannelsLastFixAugmentedNaiveDNNModelFamily):
    pass
