from abc import ABC
import keras
import keras_tuner

from models.model_container import ModelContainer


class SimpleCNNContainer(ModelContainer, ABC):
    def __init__(self, default_shape: (int, int, int)):
        super().__init__("simple_cnn", default_shape)

    def make_model(self, input_shape: (int, int, int), hyper_parameters: keras_tuner.HyperParameters) -> \
            tuple[keras.Layer, keras.Layer]:
        input_layer = keras.Input(shape=input_shape, name=self.name)

        x = self.__make_conv_layer(hyper_parameters, input_layer, default=32)
        x = self.__make_conv_layer(hyper_parameters, x, 1, default=64) \
            if hyper_parameters.Boolean(name="second_conv_layer", default=True) else x

        x = keras.layers.Flatten(data_format="channels_first")(x)

        hidden_units = hyper_parameters.Int(name="hidden_units", min_value=16, max_value=128, default=64, step=16)
        x = keras.layers.Dense(units=hidden_units, activation='relu')(x)
        output_layer = keras.layers.Dense(1, activation='sigmoid')(x)
        return input_layer, output_layer

    def __make_conv_layer(self, hyper_parameters: keras_tuner.HyperParameters,
                          previous_layer: keras.Layer, conv_id: int = 0, default: int = None) -> keras.Layer:
        kernel_size = hyper_parameters.Int(name=f"kernel_{conv_id}", min_value=2, max_value=5)
        filters = hyper_parameters.Int(name=f"filters_{conv_id}", step=16, min_value=16, max_value=64, default=default)

        x = keras.layers.Conv2D(filters=filters, kernel_size=(kernel_size, kernel_size),
                                padding='same', activation='relu', data_format='channels_first')(previous_layer)

        return keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, data_format='channels_first')(x)
