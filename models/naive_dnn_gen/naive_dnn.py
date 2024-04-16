from abc import ABC
import keras
import keras_tuner

from models.tunable_model_container import TunableModelContainer


# todo change name to something more signifcant. Container is bad
class NaiveDnnContainer(TunableModelContainer, ABC):

    def __init__(self, default_shape: (int, int, int)):
        super().__init__("naive_dnn", default_shape)

    def make_model(self, input_shape: (int, int, int), hyper_parameters: keras_tuner.HyperParameters) -> \
            tuple[keras.Layer, keras.Layer]:

        input_layer = keras.Input(shape=input_shape, name='auto_naive_dnn')
        x = keras.layers.Flatten(data_format="channels_first")(input_layer)

        for i in range(hyper_parameters.Int("layers", min_value=1, max_value=2, default=2)):
            units = hyper_parameters.Int(name=f"layer_{i}", min_value=64, max_value=1536, step=64)
            x = keras.layers.Dense(units=units)(x)

            if hyper_parameters.Boolean(name=f"dropout_{i}", default=False):
                x = keras.layers.Dropout(rate=0.25)(x)

        output_layer = keras.layers.Dense(units=1, activation='sigmoid')(x)
        return input_layer, output_layer
