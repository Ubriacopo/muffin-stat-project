import torch.utils.data
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder


def dataset_loader(image_size: tuple[int, int], data_folder_path: str = "../../data") -> \
        tuple[torchvision.datasets.ImageFolder, torchvision.datasets.ImageFolder]:
    """
    Load the dataset images in RGB after resizing to given image_size
    :param data_folder_path:
    :param image_size:
    :return:
    """
    transforms = [
        torchvision.transforms.Resize((image_size[0], image_size[1])),
        torchvision.transforms.ToTensor()
    ]

    loading_transforms = torchvision.transforms.Compose(transforms)

    train_dataset = ImageFolder(f"{data_folder_path}/train", transform=loading_transforms)
    validation_dataset = ImageFolder(f"{data_folder_path}/test", transform=loading_transforms)

    return train_dataset, validation_dataset


def dataset_information(dataset: Dataset, image_size: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns the mean and variance of the dataset.
    I followed the example shown here: https://kozodoi.me/blog/20210308/compute-image-stats
    :param image_size: The size of the images of the dataset in input
    :param dataset: Dataset to measure mean and standard deviation of
    :return: the mean and standard deviation of the dataset
    """
    sums = torch.tensor([0.0, 0.0, 0.0])
    square_sums = torch.tensor([0.0, 0.0, 0.0])

    dataloader = DataLoader(dataset, batch_size=None, num_workers=0, shuffle=False)
    size = len(dataloader) * image_size[0] * image_size[1]

    for image, _ in dataloader:
        sums += image.sum(axis=(1, 2))
        square_sums += (image ** 2).sum(axis=(1, 2))

    mean = sums / size  # Mean
    variance = square_sums / size - mean ** 2

    return mean, variance
