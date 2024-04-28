from __future__ import annotations

from typing import Final

import keras
import keras_tuner

from models.naive_dnn_gen.naive_dnn import HiddenLayerStructure
from models.structure.model_family import ModelFamily, TunableModelFamily


# todo: Move augmentation on our dataloader. I cannot find a way to work with
#       torch dataset that is CHW structure that also runs in the layers with pre-processing.
#       I suppose the tecnology is bugged or not fully porterd for these functionalities
#       Or else we use a lambda layer to reorder the input
#       https://keras.io/api/layers/reshaping_layers/reshape/
class TwoHiddenLayersAugmentedDNN(ModelFamily):

    def __init__(self):
        super().__init__("AugmentedDNN", "binary_crossentropy")
        self.hidden_layer_0 = HiddenLayerStructure(units=2048, following_dropout=0.5)
        self.hidden_layer_1 = HiddenLayerStructure(units=720, following_dropout=0.5)

        self.metrics: Final[list[str]] = ["accuracy"]

    def make_layers(self, input_shape: (int, int, int)) -> tuple[keras.Layer, keras.Layer]:

        input_layer = keras.layers.Input(shape=input_shape, name=self.name)

        # https://keras.io/api/layers/reshaping_layers/permute/
        x = keras.layers.Permute(dims=(2, 3, 1))(input_layer)  # Channels Last
        x = keras.layers.RandomFlip(mode="horizontal_and_vertical")(x)
        # x =

        x = keras.layers.Flatten()(x)

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
            model.compile(optimizer=keras.optimizers.SGD(learning_rate=1e-4, momentum=0.9, nesterov=True),
                          metrics=["accuracy"], loss=self.loss)

        model.compile(optimizer=optimizer, metrics=self.metrics, loss=self.loss)


class TwoHiddenLayersTunableAugmentedDNN(TwoHiddenLayersAugmentedDNN, TunableModelFamily):
    def load_parameters(self, hp: keras_tuner.HyperParameters):
        self.hidden_layer_0 = HiddenLayerStructure(
            hp.Int(name=f"units_0", min_value=1024, max_value=4096, step=256, default=1024),
            0.50 if hp.Boolean(name=f"dropout_0", default=False) else None
        )

        self.hidden_layer_1 = HiddenLayerStructure(
            hp.Int(name=f"units_1", min_value=128, max_value=1536, step=128, default=256),
            0.50 if hp.Boolean(name=f"dropout_1", default=False) else None
        )
