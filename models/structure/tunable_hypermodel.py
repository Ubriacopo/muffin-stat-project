import gc
from typing import Final

import keras
import keras_tuner
import torch
from keras_tuner import HyperModel
from torch.utils.data import DataLoader

from models.structure.learning_parameters.tunable_learning_parameters import \
    TunableLearningParameters
from models.structure.tunable_wrapper import TunableWrapperBase


class TunableHyperModel(HyperModel):
    def __init__(self, model_structure: TunableWrapperBase, learn_params: TunableLearningParameters,
                 input_shape: (int, int, int), tune_batch: bool = False, verbose: bool = True):
        super().__init__()

        self.input_shape: Final[(int, int, int)] = input_shape

        self.model_structure: Final[TunableWrapperBase] = model_structure
        self.learning_parameters: Final[TunableLearningParameters] = learn_params

        self.tune_batch = tune_batch
        self.verbose = verbose

    def fit(self, hp, model, *args, **kwargs):

        if not self.tune_batch:
            return super().fit(hp, model, *args, **kwargs)

        # Recreate dataloader based on choice of batch size
        # Args[0] contains the  dataloader while validation_data is in kwargs

        current_dataloader: DataLoader = args[0]
        # https://stackoverflow.com/questions/74145690/modify-args-by-decorator-in-python
        # Validation batch size is just a memory trick as
        #   ~ Since validation set is a proxy for test set, it doesn't change the validation results

        # By default, batch size of 1 will be for sure load and be maybe slower for the IO operations, but
        # I would avoid overriding another dataloader. That one will do just fine.
        new_dataloader: DataLoader = DataLoader(
            current_dataloader.dataset, shuffle=True, batch_size=hp.Choice("batch_size", values=[8, 16, 32, 64])
        )

        args = [new_dataloader]
        return model.fit(*args, batch_size=hp.Choice("batch_size", values=[8, 16, 32, 64]), **kwargs)

    def build(self, hyperparameters: keras_tuner.HyperParameters) -> keras.Model:
        # Release memory to avoid OOM during tuning.
        torch.cuda.empty_cache()
        gc.collect()

        self.model_structure.load_parameters(hyperparameters)
        self.learning_parameters.load_parameters(hyperparameters)

        model = self.model_structure.make_model(self.input_shape)

        num_params = model.count_params()

        # Real estimate is 2077966336, but we take that -10% to be sure (memory fluctuations)
        if num_params > 1870169702:
            # When this error is raised, it skips the retries as memory not sufficient
            raise keras_tuner.errors.FailedTrialError(   f"Model too large! It contains {num_params} params.")

        self.learning_parameters.compile_model(model)
        if self.verbose: model.summary()

        return model
