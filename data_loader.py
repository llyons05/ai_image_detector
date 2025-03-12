import torch
import torch.utils.data as torch_data
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import numpy as np


def load_data(train_dataset_size: int, test_dataset_size: int, train_batch_size: int) -> tuple[torch_data.DataLoader, torch_data.DataLoader]:
    """ Return (training data, testing data) """
    main_dir = "datasets"
    test_dir = f"{main_dir}/test/"
    train_dir = f"{main_dir}/train/"

    test_transform = transforms.Compose([transforms.ToTensor()])
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                            transforms.RandomVerticalFlip(p=0.5),
                                            transforms.RandomRotation(45),
                                            transforms.ToTensor()])

    test_dataset = load_dataset(test_dir, test_dataset_size, test_transform)
    train_dataset = load_dataset(train_dir, train_dataset_size, train_transform)

    test_loader = torch_data.DataLoader(test_dataset, test_dataset_size, generator=torch.Generator(device=torch.get_default_device()))
    train_loader = torch_data.DataLoader(train_dataset, train_batch_size, True, generator=torch.Generator(device=torch.get_default_device()))

    return train_loader, test_loader


def load_dataset(images_dir: str, num_images: int, transform: transforms.Compose) -> torch_data.TensorDataset:
    """ Load a dataset of all images in a given directory """
    print(f"Loading {num_images} images from {images_dir}...")

    labels_to_ids = {"REAL": torch.tensor(0), "FAKE": torch.tensor(1)}
    file_names = []
    labels = []

    for file in sorted((Path(images_dir).glob('*/*.*'))):
        label = str(file).split("\\")[-2]
        labels.append(label)
        file_names.append(file.absolute())

    index_choices = np.random.choice(np.arange(len(file_names)), num_images, replace=False).tolist() # Get n random images
    
    images = []
    ids = []
    for i in index_choices:
        image = Image.open(file_names[i])
        images.append(transform(image))
        ids.append(labels_to_ids[labels[i]])

    image_tensors = torch.stack(images, dim=0).to(torch.get_default_device())
    id_tensors = torch.stack(ids, dim=0).unsqueeze(1).float().to(torch.get_default_device())

    print("Loading complete.")
    return torch_data.TensorDataset(image_tensors, id_tensors)