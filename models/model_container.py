from __future__ import annotations

from abc import abstractmethod

import keras.optimizers


class ModelContainer:
    """
    Model that contains stuff.
    """
    @abstractmethod
    def make_model_structure(self, input_shape: (int, int, int), **kwargs):
        pass

    @abstractmethod
    def compile_model(self, optimizer: str | keras.optimizers.Optimizer, **kwargs):
        pass
