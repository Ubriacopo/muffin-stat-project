from __future__ import annotations

import gc
import re

import keras
import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision.datasets import ImageFolder

from dataset.dataset_loader import dataset_information
from models.structure.augmentation_wrapper import NormalizedDataAugmentationWrapper
from models.structure.base_model_wrapper import BaseModelWrapper
from models.structure.learning_parameters.learning_parameters import LearningParameters


class KFoldDatasetWrapper:
    def __init__(self, k: int):
        """
        The class handles K-Fold cross validation as we will use the same configuration for multiple models so
        why not implement a handy structure to make things easier?
        :param k: Number of folds. Cannot be changed but why not? It ain't hard
        """
        self.k: int = k

        self.dataset: [Subset] = []
        self.split_dataset: [Subset] = []

        self.fold_fraction: float = 0

    def load_data(self, dataset: ImageFolder):
        self.dataset = dataset

        # This way we can use automatic round-robin procedure on random_split for the splitting if
        # the dataset cannot be entirely divided by the estimated fold size.
        self.fold_fraction = 1 / self.k

        self.split_dataset = torch.utils.data.random_split(dataset, [self.fold_fraction for i in range(self.k)])

    def get_data_for_fold(self, ignored_fold: int) -> tuple[ConcatDataset, Subset]:
        """
        :param ignored_fold:
        :return: train and test datasets with current k fold selected to be the validation one
        """
        return (ConcatDataset([x for i, x in enumerate(self.split_dataset) if i != ignored_fold]),
                self.split_dataset[ignored_fold])

    def run_k_fold_cv(self, learning_parameters_template: LearningParameters,
                      # todo un metodo anche per non normalized che ricorda non è detto lo isa (modelli prefatti)
                      model_generator: BaseModelWrapper | NormalizedDataAugmentationWrapper,
                      input_shape: tuple[int, int, int], batch_size: int = 32):
        """
        :param learning_parameters_template:
        :param model_generator:
        :param input_shape: The expected input shape is C x W x H
        :param batch_size:
        :return:
        """
        test_performances = []
        test_fold_sizes = []

        for i in range(self.k):
            print(f"Starting procedure for fold {i}")
            # To avoid going OOM
            torch.cuda.empty_cache()
            gc.collect()

            train, test = self.get_data_for_fold(i)

            # This way we have 70 % Train - 10 % Validation - 20 % Test
            # Why like this? I fear that taking data away from the test split makes k fold wrong.
            train, validation = torch.utils.data.random_split(train, [0.875, 0.125])

            # Normalization step is at hand so we handle it
            if isinstance(model_generator, NormalizedDataAugmentationWrapper):
                print(f"I am calculating mean and variance of train dataset (without split {i}!")
                mean, variance = dataset_information(train, (input_shape[1], input_shape[2]))
                model_generator.load_dataset_mean_and_variance(mean, variance)

            train_dataloader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
            validation_dataloader = DataLoader(dataset=validation, batch_size=batch_size, shuffle=True)

            iteration_model = model_generator.make_model(input_shape)
            learning_parameters_template.compile_model(iteration_model)

            # Regex to turn classname to snake case (so all files follow the same rules) todo see if goes
            class_name = re.sub(r'(?<!^)(?=[A-Z])', '_', model_generator.__class__.__name__).lower()

            iteration_model.fit(train_dataloader, validation_data=validation_dataloader, epochs=80, callbacks=[
                keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='min'),
                keras.callbacks.CSVLogger(f"{class_name}_{i}.csv", separator=",", append=False)
            ])

            test_dataloader = DataLoader(dataset=test, shuffle=True)

            test_performances.append(iteration_model.evaluate(test_dataloader, verbose=1))
            test_fold_sizes.append(len(test))

        return test_performances, test_fold_sizes
