import keras
from keras import Input, Model
from keras.src.layers import Conv2D

from utils.pre_processing import AugmentationProcedure


class SimpleCNN(keras.Model):
    def __init__(self, input_shape: (int, int, int),
                 augmentation_procedure: AugmentationProcedure = None, name="naive-dnn", **kwargs):
        super().__init__(name=name, **kwargs)
        self.input_shape = input_shape

        self.augmentation_procedure = augmentation_procedure

        self.conv_layer = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same',
                                              activation='relu', input_shape=(224, 224, 3))
        self.max_pooling = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)

        self.conv_layer_2 = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same',
                                                activation='relu')
        self.max_pooling_2 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)

        self.flatten_layer = keras.layers.Flatten()
        self.hidden_layer = keras.layers.Dense(units=64, activation='relu')
        self.output_layer = keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        if self.augmentation_procedure is not None:
            inputs = self.augmentation_procedure.augmentation_procedure(inputs)
        inputs = inputs.permute(0, 2, 3, 1)
        x = self.conv_layer(inputs)
        x = self.max_pooling(x)

        x = self.conv_layer_2(x)
        x = self.max_pooling_2(x)

        x = self.flatten_layer(x)

        x = self.hidden_layer(x)
        return self.output_layer(x)

    def build_graph(self):
        """
        NOTE: Doesn't allow to delete the model
        :return:
        """
        x = Input(shape=self.input_shape)
        return Model(inputs=[x], outputs=self.call(x))
