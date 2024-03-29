from __future__ import annotations

import json


class Hyperparameters:
    def __init__(self, version: str = "1.0", learning_rate: float = 0.01, batch_size: int = 32,
                 n_epochs: int = 100, image_size: (int, int, int) = (256, 256, 3), n_classes: int = 2):
        """
        The class containing all Hyperparameters that can be met for our processes.
        :param version: Version of our parameters. Could be useless, but it's just a field, so we keep it.
        :param learning_rate: Learning rate of Model
        :param batch_size:
        :param n_epochs:
        :param image_size:
        :param n_classes:
        """
        self.version = version
        self.image_size = image_size
        self.n_epochs = n_epochs
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    @staticmethod
    def load() -> Hyperparameters:
        try:
            return Hyperparameters(**json.load(open('hyperparameters.json')))

        except FileNotFoundError:
            # Create new configuration parameters
            hp = Hyperparameters()
            Hyperparameters.save(hp)
            return hp

    @staticmethod
    def save(hp: Hyperparameters) -> Hyperparameters:
        try:
            with open("hyperparameters.json", "w") as outfile:
                outfile.write(hp.to_json())
        except Exception:
            print("Generic saving error")

        return hp
