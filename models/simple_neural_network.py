import keras
from keras import Sequential


def simple_neural_network(input_shape: (int, int, int), num_classes: int) -> Sequential:
    """
    :param input_shape: input shape
    :param num_classes: number of classes to predict
    :return: Sequential model
    """
    model = Sequential([
        keras.Input(shape=input_shape),

        keras.layers.Flatten(),

        keras.layers.Dense(units=int(1568), activation='relu'),
        keras.layers.Dense(units=int(784), activation='relu'),
        keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=keras.optimizers.SGD(learning_rate=5e-4),
                  loss='sparse_categorical_crossentropy', metrics=["accuracy"])

    return model