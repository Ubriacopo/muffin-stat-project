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


class NormalizedDataAugmentationWrapper(AugmentationWrapperBase, ABC):
    means: tuple
    variances: tuple

    def load_dataset_means_and_stds(self, dataset_means: tuple, dataset_stds: tuple):
        """
        Remember: these values are w.r.t. the training data not the whole dataset.
        :param dataset_means:
        :param dataset_stds:
        :return:
        """
        self.means = dataset_means
        self.variances = dataset_stds

    @abstractmethod
    def make_augmentation_process(self, previous_layer: keras.layers.Layer) -> keras.Layer:
        pass

    def make_augmentation(self, input_shape: (int, int, int)) -> tuple[keras.Layer, keras.Layer]:
        """
        We'd like to freeze this function as all normalized augmentations require this step.
        :param input_shape: Has to be in the shape of (channels x height x width).
        :return:
        """
        input_layer = keras.Input(shape=input_shape, name=self.__class__.__name__)

        if self.means is None or self.variances is None:
            raise Exception(f"A {self.__class__.__name__} requires to load mean and variances")

        # The torch dataloader always has axis 1 for us.
        x = self.make_augmentation_process(input_layer)

        x = keras.layers.Normalization(axis=1, mean=self.means, variance=self.variances)(x)

        return input_layer, x


class NormalizedModelWrapper(NormalizedDataAugmentationWrapper, ABC):
    def make_augmentation_process(self, previous_layer: keras.layers.Layer) -> keras.Layer:
        return previous_layer


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


class NormalizedInvertedAugmentation(NormalizedDataAugmentationWrapper, InvertedChannelsAugmentationWrapper, ABC):
    def make_augmentation_process(self, previous_layer: keras.layers.Layer) -> keras.Layer:
        x = keras.layers.Permute(dims=(2, 3, 1))(previous_layer)

        x = keras.layers.RandomFlip(mode="horizontal_and_vertical")(x)
        x = keras.layers.RandomRotation(0.3)(x)
        x = keras.layers.RandomBrightness(0.4, value_range=(-1., 1.))(x)

        return x


# todo this is the way to go. funziona
class TorchAugmentationModel(NormalizedDataAugmentationWrapper, ABC):
    def make_augmentation_process(self, previous_layer: keras.layers.Layer) -> keras.Layer:
        x = keras.layers.Permute(dims=(2, 3, 1))(previous_layer)

        x = keras.layers.RandomFlip(mode="horizontal_and_vertical")(x)
        x = keras.layers.RandomRotation(0.3)(x)
        x = keras.layers.RandomBrightness(0.4, value_range=(0., 1.))(x)

        return keras.layers.Permute(dims=(3, 2, 1))(x)
