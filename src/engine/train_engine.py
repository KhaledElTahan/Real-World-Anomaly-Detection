"""Model training engine"""

import torch
from tqdm import tqdm


@torch.enable_grad()
def train(cfg, model, loss_class, optimizer, dataloader, current_epoch, print_stats=False):
    """
    Chooses the training policy then train the model one epoch on the data loader
    Args:
        cfg (cfgNode): Model configurations
        model (torch.nn.model): Video model
        loss_class: Custom defined loss class
        optimizer (torch.nn.optimizer): The used optimizer
        dataloader (DatasetLoader): testing dataset loader
        current_epoch (int): Current epoch for the training process
        print_stats (Bool): Whether to print stats or not
    """
    assert cfg.TRAIN.TYPE in ['MIL']

    model.train()

    progress_bar = None
    if print_stats:
        progress_bar = tqdm(total=len(dataloader),
            desc="Model Trainging Progress - Epoch (" + str(current_epoch) + ")")

    if cfg.TRAIN.TYPE == "MIL":
        loss_value = multiple_instance_learning_train(
            model, loss_class, optimizer, dataloader, print_stats, progress_bar
        )

    if print_stats:
        progress_bar.close()

    if print_stats:
        print("Epoch completed with loss ", loss_value)


def multiple_instance_learning_train(model, loss_class, optimizer, dataloader, print_stats=False, progress_bar=None):
    """
    Basic multiple instance learning as in exactly as in sultani paper https://arxiv.org/abs/1801.04264v3
    Train the model one epoch on the data loader
    Args:
        model (torch.nn.model): Video model
        loss_class: Custom defined loss class
        optimizer (torch.nn.optimizer): The used optimizer
        dataloader (DatasetLoader): testing dataset loader
        current_epoch (int): Current epoch for the training process
        print_stats (Bool): Whether to print stats or not
        progress_bar (tqdm): if print_status, will be used to print a training progress bar
    Retunrs:
        loss_value (float): mean loss per example in this training epoch
    """
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

    return total_loss / len(dataloader)
