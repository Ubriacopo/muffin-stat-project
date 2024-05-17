import torch.utils.data
from torch.utils.data import ConcatDataset, DataLoader, Dataset
import torchvision

from torchvision.datasets import ImageFolder

# todo a better class for this? da rifare che sembra il cane che si morde la coda cosi
# i valori di normalization li dovrei calcolare al momento
def dataset_loader(image_size: tuple[int, int], is_grayscale: bool = False,
                   normalization_values: tuple[tuple, tuple] = None, data_folder_path: str = "../../data") -> \
        tuple[torchvision.datasets.ImageFolder, torchvision.datasets.ImageFolder]:
    """
    Load the dataset images in RGB after resizing to given image_size
    :param normalization_values:
    :param data_folder_path:
    :param is_grayscale:
    :param image_size:
    :return:
    """
    transforms = []

    if is_grayscale:
        transforms.append(torchvision.transforms.Grayscale())

    transforms.append(torchvision.transforms.Resize((image_size[0], image_size[1])))
    transforms.append(torchvision.transforms.ToTensor())

    # Normalization is either handled on the dataset loader or the Keras Model. Choice is arbitrary here.
    if normalization_values is not None:
        mean, variance = normalization_values
        transforms.append(torchvision.transforms.Normalize(mean=mean, std=variance))

    loading_transforms = torchvision.transforms.Compose(transforms)
    train_dataset = ImageFolder(f"{data_folder_path}/train", transform=loading_transforms)
    validation_dataset = ImageFolder(f"{data_folder_path}/test", transform=loading_transforms)

    return train_dataset, validation_dataset


def mean_calculator(dataset: Dataset) -> tuple[tuple, tuple]:
    """
    Diciamo molto rudimentale. Probabilmente anche lento. todo make faster
    :param dataset:
    :return:
    """
    r, g, b = [], [], []
    std_r, std_g, std_b = [], [], []

    for image, _ in DataLoader(dataset, batch_size=None, num_workers=0, shuffle=False):
        r.append(torch.mean(image[0]))
        g.append(torch.mean(image[1]))
        b.append(torch.mean(image[2]))

        std_r.append(torch.std(image[0]))
        std_g.append(torch.std(image[1]))
        std_b.append(torch.std(image[2]))

    r_mean = torch.mean(torch.stack(r))
    g_mean = torch.mean(torch.stack(g))
    b_mean = torch.mean(torch.stack(b))

    r_std = torch.std(torch.stack(r))
    g_std = torch.std(torch.stack(g))
    b_std = torch.std(torch.stack(b))

    return (r_mean, g_mean, b_mean), (r_std, g_std, b_std)
