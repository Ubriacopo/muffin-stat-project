from abc import abstractmethod, ABC

import keras
from keras.src.layers import Normalization

from models.structure.base_model_wrapper import BaseModelWrapper, Channels


# https://keras.io/examples/vision/image_classification_from_scratch/
class AugmentationWrapperBase(BaseModelWrapper):
    @abstractmethod
    def make_augmentation(self, input_shape: (int, int, int)) -> tuple[keras.Layer, keras.Layer]:
        pass

    def make_model(self, input_shape: (int, int, int)) -> keras.Model:
        """
        Creates the model by combining the augmentation procedure before the model passage.
        :param input_shape: the shape of the input. The input_shape is not allowed to change in between the
        creation of augmentation model and the actual model. If this is required the class has to be extended.
        :return: the keras model.
        """
        augmentation_input, augmentation_output = self.make_augmentation(input_shape=input_shape)
        augmentation_model = keras.Model(augmentation_input, augmentation_output, name='augmentation_model')

        input_layer, output_layer = self.make_layers(input_shape=input_shape)
        base_model = keras.Model(input_layer, output_layer, name=self.__class__.__name__)

        final_model_input = keras.Input(shape=input_shape, name=self.__class__.__name__)
        x = augmentation_model(final_model_input)

        final_model_output_layer = base_model(x)
        return keras.Model(inputs=final_model_input, outputs=final_model_output_layer)


class NormalizedDataAugmentationWrapper(AugmentationWrapperBase, ABC):
    mean: tuple
    variance: tuple

    def load_dataset_mean_and_variance(self, dataset_means: tuple, dataset_stds: tuple):
        """
        Remember: these values are w.r.t. the training data not the whole dataset.
        :param dataset_means:
        :param dataset_stds:
        :return:
        """
        self.mean = dataset_means
        self.variance = dataset_stds

    def make_model(self, input_shape: (int, int, int)) -> keras.Model:
        if self.mean is None or self.variance is None:
            raise Exception(f"A {self.__class__.__name__} requires to load mean and variances")

        augmentation_input, augmentation_output = self.make_augmentation(input_shape=input_shape)
        augmentation_output = Normalization(axis=1, mean=self.mean, variance=self.variance)(augmentation_output)

        augmentation_model = keras.Model(augmentation_input, augmentation_output)

        input_layer, output_layer = self.make_layers(input_shape=input_shape)
        base_model = keras.Model(input_layer, output_layer)

        final_model_input = keras.Input(shape=input_shape, name=self.__class__.__name__)
        x = augmentation_model(final_model_input)

        final_model_output_layer = base_model(x)
        return keras.Model(inputs=final_model_input, outputs=final_model_output_layer)


class NormalizedModelWrapper(NormalizedDataAugmentationWrapper, ABC):
    def make_augmentation(self, input_shape: (int, int, int)) -> tuple[keras.Layer, keras.Layer]:
        input_layer = keras.layers.Input(shape=input_shape, name=self.__class__.__name__)
        return input_layer, input_layer


class TorchAugmentationModel(NormalizedDataAugmentationWrapper, ABC):
    def make_augmentation(self, input_shape: (int, int, int)) -> tuple[keras.Layer, keras.Layer]:
        input_layer = keras.layers.Input(shape=input_shape, name=self.__class__.__name__)
        x = keras.layers.Permute(dims=(2, 3, 1))(input_layer)

        # Augmentation process.
        x = keras.layers.RandomFlip(mode="horizontal_and_vertical")(x)
        x = keras.layers.RandomRotation(0.3)(x)
        x = keras.layers.RandomBrightness(0.4, value_range=(0., 1.))(x)

        # Reset the shape so that the channels are first.
        x = keras.layers.Permute(dims=(3, 2, 1))(x)
        return input_layer, x
