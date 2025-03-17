from torch import nn, optim
from torch.utils import data as td

import data_loader as dl


def train_model(model: nn.Module,
                model_name: str,
                loss_fn: nn.Module,
                optimizer: optim.Optimizer,
                lr_scheduler: optim.lr_scheduler.LRScheduler,
                train_loader: td.DataLoader,
                test_loader: td.DataLoader,
                num_epochs: int,
                starting_epoch: int) -> nn.Module:
    """ Train the model for `num_epochs` epochs. """

    best_loss = 1000000
    for i in range(num_epochs):
        train_loss = train_epoch(train_loader, model, loss_fn, optimizer, len(train_loader.dataset))
        test_loss = compute_validation(test_loader, model, loss_fn, len(test_loader.dataset))
        # lr_scheduler.step(train_loss)     Removing this for now since apparently Adam is better without it

        if test_loss < best_loss:
            dl.save_model_state("best_"+ model_name, model, optimizer, lr_scheduler, i + starting_epoch)
            best_loss = test_loss

        if (i+1) % int(num_epochs/10) == 0:
            print(f"EPOCH {i+1+starting_epoch}: Train: {round(train_loss, 4)}, Test: {round(test_loss, 4)} (best loss was {round(best_loss, 4)})")
            dl.save_model_state(model_name, model, optimizer, lr_scheduler, i + starting_epoch)
    
    dl.save_model_state(model_name, model, optimizer, lr_scheduler, num_epochs + starting_epoch)
    return model
    

def train_epoch(loader: td.DataLoader, model: nn.Module, loss_fn: nn.Module, optimizer: optim.Optimizer, dataset_size: int) -> float:
    """ Train the model on the provided dataset for one epoch. Return the avg loss. """
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


def compute_validation(loader: td.DataLoader, model: nn.Module, loss_fn: nn.Module, dataset_size: int) -> float:
    """ Return the loss when tested on the validation dataset. """
    model.eval()

    total_loss = 0
    for image, target in loader:
        predictions = model(image)
        loss = loss_fn(predictions, target)
        total_loss += loss.item()
    
    return total_loss/dataset_size