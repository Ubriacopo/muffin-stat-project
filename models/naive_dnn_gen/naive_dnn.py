from __future__ import annotations

import keras
import keras_tuner

from models.structure.base_model_wrapper import BaseModelWrapper
from models.structure.layer_structure_data import HiddenLayerStructure
from models.structure.tunable_wrapper import TunableWrapperBase


class NaiveDnnWrapper(BaseModelWrapper):
    hidden_layers: list[HiddenLayerStructure] = [
        HiddenLayerStructure(units=2048, following_dropout=None),
        HiddenLayerStructure(units=1024, following_dropout=None)
    ]

    def make_layers(self, input_shape: (int, int, int)) -> tuple[keras.Layer, keras.Layer]:
        input_layer = keras.Input(shape=input_shape, name=self.__class__.__name__)
        x = keras.layers.Flatten(data_format=self.data_format.name)(input_layer)

        for layer_information in self.hidden_layers:
            x = keras.layers.Dense(units=layer_information.units, activation="relu")(x)
            x = (keras.layers.Dropout(rate=layer_information.following_dropout)(
                x)) if layer_information.following_dropout is not None else x

        output_layer = keras.layers.Dense(units=1, activation='sigmoid')(x)
        return input_layer, output_layer


class NaiveDnnTunableWrapper(NaiveDnnWrapper, TunableWrapperBase):
    def load_parameters(self, hp: keras_tuner.HyperParameters):
        self.hidden_layers = []  # Reset the default configuration.

        number_layers = hp.Int(name="layers", min_value=1, max_value=4, default=2)

        for i in range(number_layers):
            u = hp.Int(name=f"units_{i}", min_value=32, max_value=2048 - 512 * i, step=2, default=256, sampling='log')
            dropout = 0.50 if i + 1 < number_layers and hp.Boolean(name=f"dropout_{i}", default=False) else None

            self.hidden_layers.append(HiddenLayerStructure(units=u, following_dropout=dropout))


# todo: Teoricamente non è freezato sui parametri piu lontani e non posso perche altrimenti si rompe.
class NaiveDNNFinal(NaiveDnnWrapper):
    hidden_layers: tuple[HiddenLayerStructure] = (
        HiddenLayerStructure(units=21, following_dropout=None),
        HiddenLayerStructure(units=1024, following_dropout=0.5),
        HiddenLayerStructure(units=720, following_dropout=None),
    )
