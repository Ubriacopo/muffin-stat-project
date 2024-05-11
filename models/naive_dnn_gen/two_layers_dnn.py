from __future__ import annotations

from typing import Final

import keras
import keras_tuner

from models.structure.augmentation_wrapper import BasicInvertedChannelsAugmentationWrapper
from models.structure.base_model_wrapper import BaseModelWrapper
from models.structure.layer_structure_data import HiddenLayerStructure
from models.structure.tunable_wrapper import TunableWrapperBase


class TwoHiddenLayersDNNModelFamily(BaseModelWrapper):
    hidden_layer_0: HiddenLayerStructure = HiddenLayerStructure(units=2048, following_dropout=0.5)
    hidden_layer_1: HiddenLayerStructure = HiddenLayerStructure(units=720, following_dropout=0.5)

    def make_layers(self, input_shape: (int, int, int)) -> tuple[keras.Layer, keras.Layer]:
        input_layer = keras.layers.Input(shape=input_shape, name=self.__class__.__name__)
        x = keras.layers.Flatten()(input_layer)

        x = keras.layers.Dense(units=self.hidden_layer_0.units, activation="relu")(x)
        if self.hidden_layer_0.following_dropout is not None:
            x = keras.layers.Dropout(self.hidden_layer_0.following_dropout)(x)

        x = keras.layers.Dense(units=self.hidden_layer_1.units, activation="relu")(x)
        if self.hidden_layer_1.following_dropout is not None:
            x = keras.layers.Dropout(self.hidden_layer_1.following_dropout)(x)

        output_layer = keras.layers.Dense(units=1, activation='sigmoid')(x)
        return input_layer, output_layer


# todo change name
class TwoHiddenLayersTunableAugmentedDNN(TwoHiddenLayersDNNModelFamily, TunableWrapperBase):
    def load_parameters(self, hp: keras_tuner.HyperParameters):
        self.hidden_layer_0 = HiddenLayerStructure(
            hp.Int(name=f"units_0", min_value=1024, max_value=4096, step=512, default=1024),
            0.50 if hp.Boolean(name=f"dropout_0", default=False) else None
        )

        self.hidden_layer_1 = HiddenLayerStructure(
            hp.Int(name=f"units_1", min_value=128, max_value=1536, step=256, default=256),
            0.50 if hp.Boolean(name=f"dropout_1", default=False) else None
        )


class TwoHiddenLayersDNNAugModelFamily(TwoHiddenLayersTunableAugmentedDNN, BasicInvertedChannelsAugmentationWrapper):
    pass
