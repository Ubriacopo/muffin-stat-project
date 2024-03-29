import keras
from keras import Model


def build_model(input_shape: (int, int, int), model: tuple[keras.Layer, keras.Layer],
                augmentation: tuple[keras.Layer, keras.Layer] = None, name: str = "pre_processed_model") -> Model:
    """
    Build a keras model by combining optional pre-processing layers and model layers.
    If pre-processing is not defined the returned model simply runs on the given model layers.
    A function might seem overkill which I admit but at least we will be making code cleaner.
    :param model:
    :param augmentation:
    :param input_shape:
    :param name:
    :return:
    """
    base_model = Model(model[0], model[1])
    if augmentation is None:
        return base_model

    # We are sure that augmentation is well-defined, so we have to build it inside our Model.
    augmentation_model = Model(augmentation[0], augmentation[1])

    input_layer = keras.Input(shape=input_shape, name=name)
    x = augmentation_model(input_layer)
    output_layer = base_model(x)

    return Model(input_layer, output_layer)
