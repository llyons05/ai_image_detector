import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os

import data_loader as dl
from image_identifier import Image_Identifier

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

    best_loss = 1000000
    for i in range(epochs):
        train_loss = train_model(train_loader, model, bcewl_loss, optimizer, train_dataset_size)
        test_loss = compute_validation(test_loader, model, bcewl_loss, test_dataset_size)
        lr_scheduler.step(train_loss)

        if test_loss < best_loss:
            dl.save_model_state("best_model.pth", model, optimizer, lr_scheduler, i + starting_epoch)
            best_loss = test_loss

        if (i+1) % int(epochs/10) == 0:
            print(f"EPOCH {i+1+starting_epoch}: Train: {round(train_loss, 4)}, Test: {round(test_loss, 4)} (best loss was {round(best_loss, 4)})")
    

    dl.save_model_state(model_name, model, optimizer, lr_scheduler, epochs + starting_epoch)
    best_model = dl.load_existing_model_state("best_model.pth")[0]

    print()
    with torch.no_grad():
        print("Current Version:", compute_validation(test_loader, model, bcewl_loss, test_dataset_size))

    with torch.no_grad():
        print("Best Version:", compute_validation(test_loader, best_model, bcewl_loss, test_dataset_size))



def train_model(loader, model: nn.Module, loss_fn: nn.Module, optimizer: optim.Optimizer, dataset_size: int) -> float:
    """ Train the model on the provided dataset. Return the avg loss. """
    model.train()
    total_loss = 0
    for features, targets in loader:
        predictions = model(features)
        l = loss_fn(predictions, targets)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        total_loss += l.item()

    return total_loss/dataset_size


def compute_validation(loader, model: nn.Module, loss_fn: nn.Module, dataset_size: int) -> float:
    """ Return the loss when tested on the validation dataset """
    model.eval()

    total_loss = 0
    for image, target in loader:
        predictions = model(image)
        loss = loss_fn(predictions, target)
        total_loss += loss.item()
    
    return total_loss/dataset_size


def get_model_state(filename: str) -> tuple[nn.Module, optim.Optimizer, optim.lr_scheduler.LRScheduler, int]:
    if os.path.exists(f"models/{filename}"):
        return dl.load_existing_model_state(filename)

    model = Image_Identifier().to(torch.get_default_device())
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.75, patience=20)
    epoch = 0

    return model, optimizer, lr_scheduler, epoch


if __name__ == "__main__":
    torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')
    dl.ensure_all_dirs_exist()
    main()