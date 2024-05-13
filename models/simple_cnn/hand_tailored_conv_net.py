import keras
from models.structure.base_model_wrapper import BaseModelWrapper

# todo togli da qui e lascia solo su notebook?
class HandTailoredConvNetV1(BaseModelWrapper):
    def make_layers(self, input_shape: (int, int, int)) -> tuple[keras.Layer, keras.Layer]:
        input_layer = keras.Input(shape=input_shape, name=self.__class__.__name__)

        x = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same',
                                data_format=self.data_format.value, activation="relu")(input_layer)
        x = keras.layers.MaxPool2D(pool_size=(2, 2), data_format=self.data_format.value)(x)

        x = keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same',
                                data_format=self.data_format.value, activation="relu")(x)
        x = keras.layers.MaxPool2D(pool_size=(2, 2), data_format=self.data_format.value)(x)

        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(units=128, activation="relu")(x)

        output_layer = keras.layers.Dense(units=1, activation="sigmoid")(x)
        return input_layer, output_layer


