from __future__ import annotations

from abc import abstractmethod

import keras
import keras_tuner
from keras import Model


class ModelContainer(keras_tuner.HyperModel):
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
        self.augmentation = None

    def load_shape(self, shape: (int, int, int)):
        self.input_shape = shape

    def load_augmentation(self, augmentation: ModelContainer):
        self.augmentation = augmentation

    @abstractmethod
    def make_model(self, input_shape: (int, int, int), hyper_parameters: keras_tuner.HyperParameters) \
            -> tuple[keras.Layer, keras.Layer]:
        """
        Method that makes model layers.
        :param input_shape: Input shape
        :param hyper_parameters: Hyper-parameters instance
        :return: Model input and output layers. This does not build a model yet.
        """
        pass

    def build(self, hp):
        #https://github.com/keras-team/keras-tuner/issues/395
        # todo release memory?
        """
        Builds the model and compiles it. This way we are ready to train with the current set of
        hyperparameters or simply look for better ones.
        :param hyper_parameters:
        :return:
        """
        input_layer, output_layer = self.make_model(self.input_shape, hp)
        model = Model(input_layer, output_layer)

        if self.augmentation is not None:
            # todo see if this causes memory issues
            input_layer = keras.Input(shape=self.input_shape, name=self.name)
            x = self.augmentation(input_layer)
            output_layer = model(x)

            model = Model(input_layer, output_layer)

        learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
        # todo add the metric 0-1 loss
        model.compile(optimizer=keras.optimizers.Adadelta(learning_rate=learning_rate),
                      metrics=['accuracy'], loss=self.loss_function)

        return model
