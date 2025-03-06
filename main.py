import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os

from data_loader import load_data
from image_identifier import Image_Identifier

def main():
    batch_size = 75
    train_dataset_size = 20000
    test_dataset_size = 4000

    train_loader, test_loader = load_data(train_dataset_size, test_dataset_size, batch_size)
    model = get_model("ai_predictor_model.pth")
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.75, patience=20)
    bcewl_loss = nn.BCEWithLogitsLoss(reduction="sum")

    epochs = 300
    best_loss = 1000000
    for i in range(epochs):
        train_loss = train_model(train_loader, model, bcewl_loss, optimizer, train_dataset_size)
        test_loss = compute_validation(test_loader, model, bcewl_loss, test_dataset_size)

        if test_loss < best_loss:
            torch.save(model, "models/best_model.pth")
            best_loss = test_loss

        # lr_scheduler.step(train_loss)
        if (i+1) % int(epochs/10) == 0:
            print(f"EPOCH {i+1}: Train: {round(train_loss, 4)}, Test: {round(test_loss, 4)} (best loss was {round(best_loss, 4)})")
    

    torch.save(model, "models/ai_predictor_model.pth")
    best_model = torch.load("models/best_model.pth", weights_only=False)

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


def get_model(filename: str):
    if os.path.exists(f"models/{filename}"):
        model = torch.load(f"models/{filename}", weights_only=False).to(torch.get_default_device())
    else:
        # model = nn.Sequential(nn.Linear(33, 128), nn.Sigmoid(), nn.Linear(128, 64), nn.Sigmoid(), nn.Linear(64, 32), nn.Sigmoid(), nn.Linear(32, 16), nn.Sigmoid(), nn.Linear(16, 8), nn.Sigmoid(), nn.Linear(8, 4), nn.Sigmoid(), nn.Linear(4, 1))
        model = Image_Identifier().to(torch.get_default_device())

    return model


if __name__ == "__main__":
    torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')
    main()