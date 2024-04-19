import keras_tuner


class HyperBandSaveLess(keras_tuner.BayesianOptimization):
    def __init__(self, hypermodel, **kwargs):
        super().__init__(hypermodel, **kwargs)

    def run_trial(self, trial, *args, **kwargs):
        hp = trial.hyperparameters
        model = self.hypermodel.build(hp)
        return self.hypermodel.fit(hp, model, *args, **kwargs)