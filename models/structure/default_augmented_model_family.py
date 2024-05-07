from __future__ import annotations
from abc import ABC

import keras

from models.structure.augmented_model_family import InvertedChannelsAugmentedModelFamily

# todo change name
class ChannelsLastFixAugmentedNaiveDNNModelFamily(InvertedChannelsAugmentedModelFamily, ABC):

    def make_augmentation(self, input_shape: (int, int, int)) -> tuple[keras.Layer, keras.Layer]:
        input_layer = keras.Input(shape=input_shape, name=self.name)
        # https://keras.io/api/layers/reshaping_layers/permute/
        x = keras.layers.Permute(dims=(2, 3, 1))(input_layer)  # Channels Last

        x = keras.layers.RandomContrast(0.05)(x)
        x = keras.layers.RandomFlip(mode="horizontal_and_vertical")(x)

        return input_layer, x
