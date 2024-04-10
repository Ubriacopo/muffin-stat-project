import torch

from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision.datasets import ImageFolder


class KFoldController:
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


def k_fold_generator(dataset: ImageFolder, k: int):
    fold_size: int = int(len(dataset) / k)
    return torch.utils.data.random_split(dataset, [fold_size])
