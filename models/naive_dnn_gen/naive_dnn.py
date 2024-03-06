import keras


def naive_dnn_pre_process(input_shape: (int, int, int), model: keras.Model, augmentation: keras.Model) \
        -> tuple[keras.Layer, keras.Layer]:
    """
    todo documentation change name and move somehwere else
    :param input_shape:
    :param model:
    :param augmentation:
    :return:
    """
    input_layer = keras.Input(shape=input_shape, name='naive_dnn_pre')

    x = augmentation(input_layer)
    output_layer = model(x)

    return input_layer, output_layer


def naive_dnn(input_shape: (int, int, int)) -> tuple[keras.Layer, keras.Layer]:
    """
    :param input_shape:
    :return: [input, output]
    """

    input_layer = keras.Input(shape=input_shape, name='naive_dnn')
    x = keras.layers.Flatten()(input_layer)
    # Now that we do not have submodel I dunno but it works better
    # Seems like 2.9k and 1.1k are overfitting, but I ain't that sure
    x = keras.layers.Dense(units=2900, activation='relu')(x)
    x = keras.layers.Dense(units=1100, activation='relu')(x)

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
