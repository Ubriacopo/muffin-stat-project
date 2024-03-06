import json


class Hyperparameters:
    def __init__(self, version: str = "1.0", learning_rate: float = 0.01, batch_size: int = 32,
                 n_epochs: int = 100, image_size: (int, int, int) = (256, 256, 3), n_classes: int = 2):
        self.version = version
        self.image_size = image_size
        self.n_epochs = n_epochs
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)
