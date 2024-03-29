from typing import Callable

import keras


def naive_dnn(input_shape: (int, int, int)) -> tuple[keras.Layer, keras.Layer]:
    """
    :param input_shape:
    :return: [input, output]
    """

    input_layer = keras.Input(shape=input_shape, name='naive_dnn')
    x = keras.layers.Flatten(data_format="channels_first")(input_layer)

    x = keras.layers.Dense(units=700, activation='relu')(x)

    # As suggested by practical experience a 0.5 start rare should be good
    x = keras.layers.Dropout(rate=0.5)(x)

    x = keras.layers.Dense(units=250, activation='relu')(x)

    output_layer = keras.layers.Dense(units=1, activation='sigmoid')(x)
    return input_layer, output_layer


def naive_dnn_augmentation(input_shape: (int, int, int), random_rotation: float = 0.1,
                           random_flip: [bool, bool] = None) -> tuple[keras.layers.Layer, keras.layers.Layer]:
    """
    todo documentation and move somewhere else
    :param input_shape:
    :param random_rotation:
    :param random_flip:
    :return:
    """
    input_layer = keras.Input(shape=input_shape, name='naive_dnn_augmentation')
    x = keras.layers.RandomRotation(random_rotation)(input_layer)
    if random_flip is not None and random_flip[0] and random_flip[1]:
        x = keras.layers.RandomFlip("horizontal_and_vertical")(x)
    elif random_flip is not None and random_flip[0]:
        x = keras.layers.RandomFlip("horizontal")(x)
    elif random_flip is not None and random_flip[1]:
        x = keras.layers.RandomFlip("vertical")(x)

    output_layer = x
    return input_layer, output_layer
