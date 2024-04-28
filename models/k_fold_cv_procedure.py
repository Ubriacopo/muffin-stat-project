from __future__ import annotations

import gc

import keras
import torch
from torch.utils.data import DataLoader

from dataset.k_fold_dataset_wrapper import KFoldDatasetWrapper
from models.structure.model_family import ModelFamily


def k_fold_cv_procedure(model_generator: ModelFamily, input_shape: tuple[int, int, int],
                        optimizer: str | torch.optim.Optimizer, k_fold_dataset_wrapper: KFoldDatasetWrapper):
    history = []  # Contains history of each of the folds
    for i in range(k_fold_dataset_wrapper.k):
        history.append(k_fold_cv_step(model_generator, input_shape, optimizer, k_fold_dataset_wrapper, i))

    return history


def k_fold_cv_step(model_generator: ModelFamily, input_shape: tuple[int, int, int],
                   optimizer: str | torch.optim.Optimizer, k_fold_dataset_wrapper: KFoldDatasetWrapper, index: int):
    """
    Pretty straightforward implementation of K-fold cross validation step
    :param model_generator:
    :param input_shape:
    :param optimizer:
    :param k_fold_dataset_wrapper:
    :param index:
    :return:
    """
    # Release memory to avoid OOM for previous models.
    torch.cuda.empty_cache()
    gc.collect()

    train_folds, validation_fold = k_fold_dataset_wrapper.get_data_for_fold(index)
    train_dataloader = DataLoader(dataset=train_folds, batch_size=16, shuffle=True)
    validation_dataloader = DataLoader(dataset=validation_fold, batch_size=16, shuffle=True)

    iteration_model = model_generator.make_model(input_shape)
    model_generator.compile_model(iteration_model, optimizer=optimizer)

    return iteration_model.fit(
        train_dataloader, validation_data=validation_dataloader, epochs=100, verbose=1, callbacks=[
            keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=10, verbose=1, mode='min'),
            keras.callbacks.CSVLogger(f"{model_generator.name}_{index}.log", separator=",", append=False)
        ])
