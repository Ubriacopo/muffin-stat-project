import torchvision


def dataset_loader(image_size: tuple[int, int], is_grayscale: bool = False,
                   normalization: bool = False, data_folder_path: str = "../../data") -> \
        tuple[torchvision.datasets.ImageFolder, torchvision.datasets.ImageFolder]:
    """
    Load the dataset images in RGB after resizing to given image_size
    :param data_folder_path:
    :param is_grayscale:
    :param image_size:
    :return:
    """
    transforms = []
    transforms.append(torchvision.transforms.Grayscale()) if is_grayscale else None
    transforms.append(torchvision.transforms.Resize((image_size[0], image_size[1])))
    transforms.append(torchvision.transforms.ToTensor())

    if normalization:
        transforms.append(torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    loading_transforms = torchvision.transforms.Compose(transforms)
    train_dataset = torchvision.datasets.ImageFolder(f"{data_folder_path}/train", transform=loading_transforms)
    validation_dataset = torchvision.datasets.ImageFolder(f"{data_folder_path}/test", transform=loading_transforms)

    return train_dataset, validation_dataset
