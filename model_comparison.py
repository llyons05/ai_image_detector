import torch
from torch import nn
from torchvision.transforms import transforms

import data_loader as dl
from model_trainer import compute_validation


def main():
    model_1_name = input("Model 1 name: ")
    model_1 = dl.load_existing_model_state(model_1_name)[0]

    model_2_name = input("Model 2 name: ")
    model_2 = dl.load_existing_model_state(model_2_name)[0]

    validation_size = int(input("Number of images to test on: "))
    validation_loader = load_validation_dataset(validation_size)

    loss_fn = nn.BCEWithLogitsLoss(reduction="sum")

    losses, accuracies = compare_models(model_1, model_2, validation_loader, loss_fn)
    print_results((model_1_name, model_2_name), losses, accuracies)



def load_validation_dataset(num_images: int) -> torch.utils.data.DataLoader:
    val_transform = transforms.Compose([transforms.ToTensor()])
    dataset = dl.load_dataset(dl.TEST_DIR, num_images, val_transform)
    return torch.utils.data.DataLoader(dataset, num_images, generator=torch.Generator(device=torch.get_default_device()))


def compare_models(model_1: nn.Module, model_2: nn.Module, data_loader: torch.utils.data.DataLoader, loss_fn: nn.Module) -> tuple[tuple[float, float], tuple[float, float]]:
    """ Return `(model_1 loss, model_2 loss), (model_1 acc, model_2 acc)`. """
    model_1_loss, model_2_loss = 0, 0
    model_1_acc, model_2_acc = 0, 0
    with torch.no_grad():
        model_1_loss = compute_validation(data_loader, model_1, loss_fn, len(data_loader.dataset))
        model_2_loss = compute_validation(data_loader, model_2, loss_fn, len(data_loader.dataset))
        model_1_acc = get_accuracy(model_1, data_loader)
        model_2_acc = get_accuracy(model_2, data_loader)
    
    return (model_1_loss, model_2_loss), (model_1_acc, model_2_acc)


def get_accuracy(model: nn.Module, data_loader: torch.utils.data.DataLoader) -> float:
    """ After getting the model's prediction, round it to 0 or 1. Then return the loss. """
    model.eval()

    total_misses = 0
    for image, target in data_loader:
        predictions = model(image)
        predictions = torch.nn.functional.sigmoid(predictions)
        predictions = torch.round(predictions)
        loss = torch.nn.functional.l1_loss(predictions, target, reduction='sum')
        total_misses += loss.item()
    
    return 1 - total_misses/len(data_loader.dataset)


def print_results(names: tuple[str, str], losses: tuple[float, float], accuracies: tuple[float, float]) -> None:
    print("\nRESULTS:")
    max_len = max(len(name) for name in names) + 2

    for i in range(len(names)):
        string = "\t" + f"{names[i]}: ".rjust(max_len)
        string += "Loss: %.4f (%0+.4f)\t" % (losses[i], losses[i] - losses[(i+1)%len(names)])
        string += "Accuracy: %.4f (%0+.4f)" % (accuracies[i], accuracies[i] - accuracies[(i+1)%len(names)])
        print(string)
    print()


if __name__ == "__main__":
    main()