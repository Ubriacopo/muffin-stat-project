from abc import abstractmethod, ABC

import keras

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
        augmentation_model = keras.Model(augmentation_input, augmentation_output)

        input_layer, output_layer = self.make_layers(input_shape=input_shape)
        base_model = keras.Model(input_layer, output_layer)

        final_model_input = keras.Input(shape=input_shape, name=self.__class__.__name__)
        x = augmentation_model(final_model_input)

        final_model_output_layer = base_model(x)
        return keras.Model(inputs=final_model_input, outputs=final_model_output_layer)


class InvertedChannelsAugmentationWrapper(AugmentationWrapperBase, ABC):

    def __init__(self):
        super().__init__(data_format=Channels.channels_last)

    def make_model(self, input_shape: (int, int, int)) -> keras.Model:
        C, W, H = input_shape  # Channels / Width / Height

        augmentation_input, augmentation_output = self.make_augmentation(input_shape=(C, W, H))
        augmentation_model = keras.Model(augmentation_input, augmentation_output)

        input_layer, output_layer = self.make_layers(input_shape=(W, H, C))
        base_model = keras.Model(input_layer, output_layer)

        final_model_input = keras.Input(shape=input_shape, name=self.__class__.__name__)
        x = augmentation_model(final_model_input)
        final_model_output_layer = base_model(x)
        return keras.Model(inputs=final_model_input, outputs=final_model_output_layer)


class CustomInvertedAugmentationWrapper(InvertedChannelsAugmentationWrapper, ABC):

    def make_augmentation(self, input_shape: (int, int, int)) -> tuple[keras.Layer, keras.Layer]:
        input_layer = keras.Input(shape=input_shape, name=self.__class__.__name__)
        x = keras.layers.Permute(dims=(2, 3, 1))(input_layer)  # Channels Last

        x = keras.layers.RandomFlip(mode="horizontal_and_vertical")(x)
        x = keras.layers.RandomRotation(0.3)(x)
        x = keras.layers.RandomBrightness(0.4, value_range=(0., 1.))(x)

        return input_layer, x


class NormalizedDataAugmentationWrapper(AugmentationWrapperBase, ABC):
    means: tuple
    variances: tuple

    # The calculated means and stds of the TRAINING SET: IMPORTANTE GUARAD:

    # Common pitfall. An important point to make about the preprocessing is that any preprocessing statistics
    # (e.g. the data mean) must only be computed on the training data, and then applied to the validation / test data.
    # E.g. computing the mean and subtracting it from every image across the entire dataset and then splitting
    # the data into train/val/test splits would be a mistake. Instead, the mean must be computed only over the
    # training data and then subtracted equally from all splits (train/val/test).
    def load_dataset_means_and_stds(self, dataset_means: tuple, dataset_stds: tuple):
        self.means = dataset_means
        self.variances = dataset_stds

    def make_augmentation(self, input_shape: (int, int, int)) -> tuple[keras.Layer, keras.Layer]:
        input_layer = keras.Input(shape=input_shape, name=self.__class__.__name__)

        axis = 3 if self.data_format.value is Channels.channels_last else 1
        x = keras.layers.Normalization(axis=axis, mean=self.means, variance=self.variances)(input_layer)

        return input_layer, x


class NormalizedInvertedAugmentation(NormalizedDataAugmentationWrapper, InvertedChannelsAugmentationWrapper, ABC):

    def make_augmentation(self, input_shape: (int, int, int)) -> tuple[keras.Layer, keras.Layer]:
        input_layer = keras.Input(shape=input_shape, name=self.__class__.__name__)
        # Invert
        x = keras.layers.Permute(dims=(2, 3, 1))(input_layer)

        # Normalize the data
        axis = 1 if self.data_format.value is Channels.channels_last else 3
        x = keras.layers.Normalization(axis=axis, mean=self.means, variance=self.variances)(x)

        # Make augmentations (Disabled in validation)
        x = keras.layers.RandomFlip(mode="horizontal_and_vertical")(x)
        x = keras.layers.RandomRotation(0.3)(x)
        x = keras.layers.RandomBrightness(0.4, value_range=(-1., 1.))(x)

        return input_layer, x
