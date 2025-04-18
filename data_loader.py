import torch
import torch.utils.data as torch_data
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import numpy as np
import os
import json


MODEL_DIR = "models"
DATA_DIR = "datasets"
FIGURE_DIR = "figures"
TEST_DIR = f"{DATA_DIR}/test/"
TRAIN_DIR = f"{DATA_DIR}/train/"


def load_data(train_dataset_size: int, test_dataset_size: int, train_batch_size: int) -> tuple[torch_data.DataLoader, torch_data.DataLoader]:
    """ Return (training data, testing data) """

    test_transform = transforms.Compose([transforms.ToTensor()])
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                            transforms.RandomVerticalFlip(p=0.5),
                                            transforms.RandomRotation(45),
                                            transforms.ToTensor()])

    test_dataset = load_dataset(TEST_DIR, test_dataset_size, test_transform)
    train_dataset = load_dataset(TRAIN_DIR, train_dataset_size, train_transform)

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


def load_existing_model_state(model_name: str) -> tuple[torch.nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler, int]:
    data_dict = torch.load(f"models/{model_name}.pth", weights_only=False)
    model = data_dict["model"].to(torch.get_default_device())
    optimizer = data_dict["optimizer"]
    scheduler = data_dict["scheduler"]
    epoch = data_dict["epoch"]

    return model, optimizer, scheduler, epoch


def save_model_state(model_name: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler, epoch: int) -> None:
    data_dict = {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "epoch": epoch
    }

    torch.save(data_dict, f"models/{model_name}.pth")


def ensure_all_dirs_exist() -> None:
    for folder in [FIGURE_DIR, MODEL_DIR, DATA_DIR]:
        if not os.path.exists(folder):
            print(f"Setting up directory: {folder}")
            os.makedirs(folder)


def load_config(config_filename: str = "config.json") -> dict:
    if not os.path.exists(config_filename):
        raise Exception("Invalid config filename. Please make a config.json with 'model_name', 'num_epochs', 'train_size', 'test_size', and 'batch_size'.")
    
    with open(config_filename) as config_file:
        config = json.load(config_file)
    return config


if __name__ == "__main__":
    ensure_all_dirs_exist()