import keras

import models.structure.base_model_family


class HandTailoredConvNetV1(models.structure.base_model_family.BaseModelFamily):
    def __init__(self):
        super().__init__("HandmadeConvNet", "binary_crossentropy")

    def make_layers(self, input_shape: (int, int, int)) -> tuple[keras.Layer, keras.Layer]:
        input_layer = keras.Input(shape=input_shape, name=self.name)
        x = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same',
                                data_format=self.data_format.value, activation="relu")(input_layer)

        x = keras.layers.MaxPool2D(pool_size=(2, 2), data_format=self.data_format.value)(x)

        x = keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same',
                                data_format=self.data_format.value, activation="relu")(input_layer)
        x = keras.layers.MaxPool2D(pool_size=(2, 2), data_format=self.data_format.value)(x)

        x = keras.layers.Flatten()
        x = keras.layers.Dense(units=128, activation="relu")(x)

        output_layer = keras.layers.Dense(units=1, activation="sigmoid")(x)
        return input_layer, output_layer