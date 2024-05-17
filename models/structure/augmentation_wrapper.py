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


class BasicInvertedChannelsAugmentationWrapper(InvertedChannelsAugmentationWrapper, ABC):
    def make_augmentation(self, input_shape: (int, int, int)) -> tuple[keras.Layer, keras.Layer]:
        input_layer = keras.Input(shape=input_shape, name=self.__class__.__name__)
        # https://keras.io/api/layers/reshaping_layers/permute/
        x = keras.layers.Permute(dims=(2, 3, 1))(input_layer)  # Channels Last

        x = keras.layers.RandomContrast(0.05)(x)
        x = keras.layers.RandomFlip(mode="horizontal_and_vertical")(x)

        return input_layer, x


class InvertedAugmentationWrapper(InvertedChannelsAugmentationWrapper, ABC):
    def make_augmentation(self, input_shape: (int, int, int)) -> tuple[keras.Layer, keras.Layer]:
        input_layer = keras.Input(shape=input_shape, name=self.__class__.__name__)
        x = keras.layers.Permute(dims=(2, 3, 1))(input_layer)  # Channels Last

        # x = keras.layers.RandomBrightness(0.2)(x)
        x = keras.layers.RandomFlip(mode="horizontal_and_vertical")(x)
        x = keras.layers.RandomRotation(0.3)(x)

        return input_layer, x


class CustomInvertedAugmentationWrapper(InvertedChannelsAugmentationWrapper, ABC):
    def make_augmentation(self, input_shape: (int, int, int)) -> tuple[keras.Layer, keras.Layer]:
        input_layer = keras.Input(shape=input_shape, name=self.__class__.__name__)
        x = keras.layers.Permute(dims=(2, 3, 1))(input_layer)  # Channels Last

        x = keras.layers.RandomFlip(mode="horizontal_and_vertical")(x)
        x = keras.layers.RandomRotation(0.3)(x)
        # Input is already normalized in [0,1]
        x = keras.layers.RandomBrightness(0.4, value_range=(0., 1.))(x)

        return input_layer, x