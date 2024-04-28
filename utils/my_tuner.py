import glob
import keras_tuner
import os


class HistoryDeletingBayesianOptimization(keras_tuner.BayesianOptimization):
    """
    Deletes the model generated during the tuning process as we only care about the parameters
    selected and the evalutation of the model. The best ones will be retrained.
    """

    def __init__(self, hypermodel, **kwargs):
        super().__init__(hypermodel, **kwargs)

    def run_trial(self, trial, *args, **kwargs):
        # Remove previous iterations models if there are any
        for filename in glob.iglob(f'{self.directory}/**/*.h5', recursive=True):
            os.remove(filename)

        return super().run_trial(trial, *args, **kwargs)
