import keras
from keras.src import Functional

from models.structure.base_model_wrapper import BaseModelWrapper


class XceptionAugmented(BaseModelWrapper):
    latest_xception_model: Functional

    def make_layers(self, input_shape: (int, int, int)) -> tuple[keras.Layer, keras.Layer]:
        C, W, H = input_shape

        inputs = keras.Input(input_shape)

        x = keras.layers.Permute((2, 3, 1))(inputs)

        # Augmentation process.
        x = keras.layers.RandomFlip(mode="horizontal_and_vertical")(x)
        x = keras.layers.RandomRotation(0.3)(x)
        x = keras.layers.RandomBrightness(0.4, value_range=(0., 1.))(x)

        x = keras.layers.Rescaling(255)(x)
        x = keras.applications.xception.preprocess_input(x)
        self.latest_xception_model = keras.applications.Xception(
            weights='imagenet', include_top=False, input_shape=(W, H, C)
        )

        self.latest_xception_model.trainable = False
        x = self.latest_xception_model(x, training=False)
        x = keras.layers.GlobalAveragePooling2D()(x)

        outputs = keras.layers.Dense(1, activation='sigmoid')(x)
        return inputs, outputs


class VGG16Custom(BaseModelWrapper):
    latest_model: Functional

    def make_layers(self, input_shape: (int, int, int)) -> tuple[keras.Layer, keras.Layer]:
        C, W, H = input_shape

        inputs = keras.Input(input_shape)
        x = keras.layers.Permute((2, 3, 1))(inputs)

        # Augmentation process.
        x = keras.layers.RandomFlip(mode="horizontal_and_vertical")(x)
        x = keras.layers.RandomRotation(0.3)(x)
        x = keras.layers.RandomBrightness(0.4, value_range=(0., 1.))(x)

        x = keras.layers.Rescaling(255)(x)  # Avoid torch problem
        x = keras.applications.vgg16.preprocess_input(x)

        self.latest_model = keras.applications.vgg16.VGG16(
            weights='imagenet', include_top=False, input_shape=(W, H, C)
        )

        self.latest_model.trainable = False
        x = self.latest_model(x, training=False)
        x = keras.layers.GlobalAveragePooling2D()(x)

        outputs = keras.layers.Dense(1, activation='sigmoid')(x)
        return inputs, outputs
