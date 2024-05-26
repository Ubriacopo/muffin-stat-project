from abc import abstractmethod

import keras_tuner

from models.structure.base_model_wrapper import BaseModelWrapper

# todo cambia in classe a se visto che è identica a tunable_learning_parameters
class TunableWrapperBase(BaseModelWrapper):
    """
    A tunable model wrapper allows for the creation of models with keras tuner hyperparameters.
    (In python it actually behaves like an interface)
    """

    @abstractmethod
    def load_parameters(self, hyperparameters: keras_tuner.HyperParameters):
        """
        Load the network structure parameters from the hyperparameters.
        :param hyperparameters: instance of hyperparameters that define the search space
        """
        pass
