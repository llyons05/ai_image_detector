import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os

import data_loader as dl
import model_trainer as trainer
from image_identifier import Image_Identifier
from model_comparison import compare_models, print_results


def main():
    config = dl.load_config()
    model_name = config["model_name"]
    epochs = config["num_epochs"]
    batch_size = config["batch_size"]
    train_dataset_size = config["train_size"]
    test_dataset_size = config["test_size"]

    train_loader, test_loader = dl.load_data(train_dataset_size, test_dataset_size, batch_size)
    model, optimizer, lr_scheduler, starting_epoch = get_model_state(model_name)
    bcewl_loss = nn.BCEWithLogitsLoss(reduction="sum")

    model = trainer.train_model(model, model_name, bcewl_loss, optimizer, lr_scheduler, train_loader, test_loader, epochs, starting_epoch)
    best_model = dl.load_existing_model_state("best_"+ model_name)[0]

    losses, accuracies = compare_models(model, best_model, test_loader, bcewl_loss)
    print_results((model_name, "best_"+ model_name), losses, accuracies)


def get_model_state(model_name: str) -> tuple[nn.Module, optim.Optimizer, optim.lr_scheduler.LRScheduler, int]:
    if os.path.exists(f"models/{model_name}.pth"):
        return dl.load_existing_model_state(model_name)

    model = Image_Identifier().to(torch.get_default_device())
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.75, patience=20)
    epoch = 0

    return model, optimizer, lr_scheduler, epoch


if __name__ == "__main__":
    torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')
    dl.ensure_all_dirs_exist()
    main()