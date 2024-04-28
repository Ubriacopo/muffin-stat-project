from __future__ import annotations

from abc import abstractmethod
from typing import Final

import keras
import keras_tuner


class ModelFamily:
    """"
    todo: Change name?
    """

    def __init__(self, family_name: str, loss: str):
        self.name: Final[str] = family_name
        self.loss: Final[str] = loss

    @abstractmethod
    def make_layers(self, input_shape: (int, int, int)) -> tuple[keras.Layer, keras.Layer]:
        """
        It defines the main structure of the model.
        :param input_shape:
        :return:
        """
        pass

    @abstractmethod
    def compile_model(self, model: keras.Model, optimizer: str | keras.optimizers.Optimizer | None):
        pass

    def make_model(self, input_shape: (int, int, int)) -> keras.Model:
        input_layer, output_layer = self.make_layers(input_shape=input_shape)
        return keras.Model(inputs=input_layer, outputs=output_layer)


class AugmentedModelFamily(ModelFamily):

    @abstractmethod
    def make_augmentation(self, input_shape: (int, int, int)):
        pass

    def make_model(self, input_shape: (int, int, int)) -> keras.Model:
        augmentation_input, augmentation_output = self.make_augmentation(input_shape=input_shape)
        augmentation_model = keras.Model(augmentation_input, augmentation_output)

        input_layer, output_layer = self.make_layers(input_shape=input_shape)
        base_model = keras.Model(input_layer, output_layer)

        final_model_input = keras.Input(shape=input_shape, name=self.name)
        x = augmentation_model(final_model_input)
        final_model_output_layer = base_model(x)
        return keras.Model(inputs=final_model_input, outputs=final_model_output_layer)


class TunableModelFamily(ModelFamily):
    @abstractmethod
    def load_parameters(self, hyperparameters: keras_tuner.HyperParameters):
        pass


class FinalModelFamilyInstance(ModelFamily):
    @abstractmethod
    def get_model(self, input_shape: (int, int, int), default_compile: bool = True) -> keras.Model:
        pass
