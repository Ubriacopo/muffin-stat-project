import torch
import torchvision


def dataset_loader(image_size: tuple[int, int], is_grayscale: bool = False,
                   data_folder_path: str = "../../data") -> \
        tuple[torchvision.datasets.ImageFolder, torchvision.datasets.ImageFolder]:
    """
    Load the dataset images in RGB after resizing to given image_size
    :param is_grayscale:
    :param image_size:
    :return:
    """
    transforms = []
    transforms.append(torchvision.transforms.Grayscale()) if is_grayscale else None
    transforms.append(torchvision.transforms.Resize((image_size[0], image_size[1])))
    transforms.append(torchvision.transforms.ToTensor())

    loading_transforms = torchvision.transforms.Compose(transforms)
    train_dataset = torchvision.datasets.ImageFolder(f"{data_folder_path}/train", transform=loading_transforms)
    validation_dataset = torchvision.datasets.ImageFolder(f"{data_folder_path}/test", transform=loading_transforms)

    return train_dataset, validation_dataset
