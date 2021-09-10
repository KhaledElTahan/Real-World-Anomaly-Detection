"""Model training engine"""

import torch
from tqdm import tqdm


@torch.enable_grad()
def train(model, loss_class, optimizer, dataloader, current_epoch, print_stats=False):
    """
    Train the model one epoch on the data loader
    Args:
        model (torch.nn.model): Video model
        loss_class: Custom defined loss class
        optimizer (torch.nn.optimizer): The used optimizer
        dataloader (DatasetLoader): testing dataset loader
        current_epoch (int): Current epoch for the training process
        print_stats (Bool): Whether to print stats or not
    """
    model.train()

    if print_stats:
        progress_bar = tqdm(total=len(dataloader),
            desc="Model Trainging Progress - Epoch (" + str(current_epoch) + ")")

    total_loss = 0.0

    for normal_batch, anomaly_batch in dataloader:
        normal_preds = model(normal_batch["features_batched"])
        anomaly_preds = model(anomaly_batch["features_batched"])

        loss = loss_class(normal_preds, anomaly_preds)
        our_loss = loss()

        total_loss += our_loss.detach().item()

        optimizer.zero_grad()
        our_loss.backward()
        optimizer.step()

        if print_stats:
            progress_bar.update(n=1)

    if print_stats:
        progress_bar.close()

    if print_stats:
        print("Epoch completed with loss ", total_loss / len(dataloader))
