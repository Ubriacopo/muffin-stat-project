from abc import ABC
import keras
import keras_tuner

from models.tunable_model_container import TunableModelContainer


# Refactor like this https://keras.io/guides/keras_tuner/getting_started/#keep-keras-code-separate
# https://d2l.ai/chapter_convolutional-neural-networks/padding-and-strides.html
class SimpleCNNContainer(TunableModelContainer, ABC):
    def __init__(self, default_shape: (int, int, int)):
        super().__init__("simple_cnn", default_shape)

    def make_model_structure(self, input_shape: (int, int, int), conv_layers: [tuple], hidden_units) \
            -> tuple[keras.Layer, keras.Layer]:
        x = None  # The variable is used but not yet initialized
        input_layer = keras.Input(shape=input_shape, name=self.name)
        for kernel_size, filters in conv_layers:
            x = self.__make_conv_layer(kernel_size, filters, input_layer if x is None else x)
        x = keras.layers.Flatten(data_format="channels_first")(x)

        x = keras.layers.Dense(units=hidden_units, activation='relu')(x)
        output_layer = keras.layers.Dense(1, activation='sigmoid')(x)

        return input_layer, output_layer

    def make_model_with_hyperparameters(self, input_shape: (int, int, int),
                                        hyper_parameters: keras_tuner.HyperParameters) -> \
            tuple[keras.Layer, keras.Layer]:
        return self.make_model_structure(
            input_shape,
            conv_layers=[[
                hyper_parameters.Int(name=f"kernel_0", min_value=2, max_value=5),
                hyper_parameters.Int(name=f"filters_0", step=16, min_value=16, max_value=64)
            ], [
                hyper_parameters.Int(name=f"kernel_1", min_value=2, max_value=5),
                hyper_parameters.Int(name=f"filters_1", step=16, min_value=16, max_value=64)
            ]],
            hidden_units=hyper_parameters.Int(name="hidden_units", min_value=16, max_value=128, default=64, step=16)
        )

    def __make_conv_layer(self, kernel_size, filters, previous_layer: keras.Layer) -> keras.Layer:
        x = keras.layers.Conv2D(filters=filters, kernel_size=(kernel_size, kernel_size),
                                padding='same', activation='relu', data_format='channels_first')(previous_layer)
        return keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, data_format='channels_first')(x)
