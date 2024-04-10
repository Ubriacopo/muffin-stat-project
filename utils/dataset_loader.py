import torch
import torchvision


def dataset_loader(image_size: tuple[int, int]) -> \
        tuple[torchvision.datasets.ImageFolder, torchvision.datasets.ImageFolder]:
    loading_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((image_size[0], image_size[1])), torchvision.transforms.ToTensor()
    ])

    train_dataset = torchvision.datasets.ImageFolder("../../data/train", transform=loading_transforms)
    validation_dataset = torchvision.datasets.ImageFolder("../../data/test/", transform=loading_transforms)

    return train_dataset, validation_dataset
