from __future__ import annotations
from abc import abstractmethod, ABC

import keras

from models.structure.base_model_family import BaseModelFamily


class BaseAugmentedModelFamily(BaseModelFamily):
    @abstractmethod
    def make_augmentation(self, input_shape: (int, int, int)) -> tuple[keras.Layer, keras.Layer]:
        pass

    @abstractmethod
    def make_model(self, input_shape: (int, int, int)) -> keras.Model:
        pass


class InvertedChannelsAugmentedModelFamily(BaseAugmentedModelFamily, ABC):

    def make_model(self, input_shape: (int, int, int)) -> keras.Model:
        C, W, H = input_shape  # Channels / Width / Height

        augmentation_input, augmentation_output = self.make_augmentation(input_shape=(C, W, H))
        augmentation_model = keras.Model(augmentation_input, augmentation_output)

        input_layer, output_layer = self.make_layers(input_shape=(W, H, C))
        base_model = keras.Model(input_layer, output_layer)

        final_model_input = keras.Input(shape=input_shape, name=self.name)
        x = augmentation_model(final_model_input)
        final_model_output_layer = base_model(x)
        return keras.Model(inputs=final_model_input, outputs=final_model_output_layer)
