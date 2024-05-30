from models.structure.augmentation_wrapper import NormalizedModelWrapper, TorchAugmentationModel
from keras.layers import Conv2D, MaxPool2D, Input, Flatten, Dense, Layer


class HandTailoredConvNet(NormalizedModelWrapper):
    """
    First attempt on building a CNN.
    ((CONV -> POOL) x 2 ) -> FLATTEN -> DENSE -> DENSE
    """

    def make_layers(self, input_shape: (int, int, int)) -> tuple[Layer, Layer]:
        chan = self.data_format.value
        input_layer = Input(shape=input_shape, name=self.__class__.__name__)

        x = Conv2D(64, kernel_size=(3, 3), padding='same', data_format=chan, activation="relu")(input_layer)
        x = MaxPool2D(pool_size=(2, 2), data_format=chan)(x)

        x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', data_format=chan, activation="relu")(x)
        x = MaxPool2D(pool_size=(2, 2), data_format=chan)(x)

        x = Flatten(data_format=chan)(x)
        x = Dense(units=128, activation="relu")(x)

        output_layer = Dense(units=1, activation="sigmoid")(x)
        return input_layer, output_layer


class AugmentedHandTailoredConvNet(TorchAugmentationModel, HandTailoredConvNet):
    pass


class SmallerHandTailoredConvNet(TorchAugmentationModel):
    """
    Second attempt. Reduce the parameters to avoid overfitting.
    We have fewer filters and the first conv layer has a wider mask meaning
    it has a harder time memorizing small patterns in the first place.
    """

    def make_layers(self, input_shape: (int, int, int)) -> tuple[Layer, Layer]:
        chan = self.data_format.value
        input_layer = Input(shape=input_shape, name=self.__class__.__name__)

        x = Conv2D(64, kernel_size=(5, 5), padding='same', data_format=chan, activation="relu")(input_layer)
        x = MaxPool2D(pool_size=(2, 2), data_format=chan)(x)

        x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', data_format=chan, activation="relu")(x)
        x = MaxPool2D(pool_size=(2, 2), data_format=chan)(x)

        x = Flatten(data_format=chan)(x)
        x = Dense(units=128, activation="relu")(x)

        output_layer = Dense(units=1, activation="sigmoid")(x)
        return input_layer, output_layer
