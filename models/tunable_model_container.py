from __future__ import annotations

import gc
from abc import abstractmethod

import keras
import keras_tuner
import torch
from keras import Model

from models.model_container import ModelContainer


class TunableModelContainer(keras_tuner.HyperModel, ModelContainer):
    def __init__(self, name: str, default_shape: (int, int, int), loss_function: str = "binary_crossentropy"):
        """
        A ModelContainer is simply a structure that wraps the construction of the model to use the hyperparameters
        of learning correctly. This is to force the structure of our model to be always the same.
        :param name: Name of the model.
        :param default_shape: Default shape of the input data.
        """
        super().__init__()

        self.input_shape = default_shape
        self.loss_function = loss_function

        self.name = name
        self.augmentation: TunableModelContainer | None = None


    @abstractmethod
    def make_model_with_hyperparameters(self, input_shape: (int, int, int),
                                        hyperparameters: keras_tuner.HyperParameters) \
            -> tuple[keras.Layer, keras.Layer]:
        """
        Method that makes model layers.
        :param input_shape: Input shape
        :param hyperparameters: Hyper-parameters instance
        :return: Model input and output layers. This does not build a model yet.
        """
        pass

    def build(self, hp):
        # https://github.com/keras-team/keras-tuner/issues/395
        # todo release memory?
        torch.cuda.empty_cache()
        gc.collect()

        input_layer, output_layer = self.make_model_with_hyperparameters(self.input_shape, hp)
        model = Model(input_layer, output_layer)

        if self.augmentation is not None:
            # todo see if this causes memory issues
            input_layer = keras.Input(shape=self.input_shape, name=self.name)
            x = self.augmentation(input_layer)
            output_layer = model(x)

            model = Model(input_layer, output_layer)

        self.compile_model(model, hp)
        return model
