"""Model training engine"""

import torch
from tqdm import tqdm

from src.engine import test_engine


@torch.enable_grad()
def train(cfg, model, loss_class, optimizer, train_dataloader, test_dataloader, current_epoch, print_stats=False):
    """
    Chooses the training policy then train the model one epoch on the data loader
    Args:
        cfg (cfgNode): Model configurations
        model (torch.nn.model): Video model
        loss_class: Custom defined loss class
        optimizer (torch.nn.optimizer): The used optimizer
        train_dataloader (DatasetLoader): training dataset loader
        test_dataloader (DatasetLoader): testing dataset loader
        current_epoch (int): Current epoch for the training process
        print_stats (Bool): Whether to print stats or not
    Returns:
        loss_value (float): The loss of one epoch
    """
    assert cfg.TRAIN.TYPE in ['MIL']

    model.train()

    progress_bar = None
    if print_stats:
        progress_bar = tqdm(
            total=len(train_dataloader),
            desc="Trainging Progress - Epoch (" + str(current_epoch) + ")",
            unit="batch",
            colour="green"
        )

    if cfg.TRAIN.TYPE == "MIL":
        loss_value = multiple_instance_learning_train(
            cfg, model, loss_class, optimizer, train_dataloader, test_dataloader, print_stats, progress_bar
        )

    if print_stats:
        progress_bar.close()

    return loss_value


def multiple_instance_learning_train(cfg, model, loss_class, optimizer, train_dataloader, test_dataloader, print_stats=False, progress_bar=None):
    """
    Basic multiple instance learning as in exactly as in sultani paper https://arxiv.org/abs/1801.04264v3
    Train the model one epoch on the data loader
    Args:
        cfg (cfgNode): Model configurations
        model (torch.nn.model): Video model
        loss_class: Custom defined loss class
        optimizer (torch.nn.optimizer): The used optimizer
        train_dataloader (DatasetLoader): training dataset loader
        test_dataloader (DatasetLoader): testing dataset loader
        current_epoch (int): Current epoch for the training process
        print_stats (Bool): Whether to print stats or not
        progress_bar (tqdm): if print_status, will be used to print a training progress bar
    Retunrs:
        loss_value (float): mean loss per example in this training epoch
    """
    total_loss = 0.0

    for idx, (normal_batch, anomaly_batch) in enumerate(train_dataloader):
        features_normal_batch = normal_batch["features_batched"]
        features_anomaly_batch = anomaly_batch["features_batched"]

        if cfg.NUM_GPUS > 0:
            features_normal_batch = features_normal_batch.cuda()
            features_anomaly_batch = features_anomaly_batch.cuda()

        normal_preds = model(features_normal_batch)
        anomaly_preds = model(features_anomaly_batch)

        loss = loss_class(normal_preds, anomaly_preds)
        our_loss = loss()

        assert our_loss.requires_grad # to make sure testing inside training doesn't cause problems

        total_loss += our_loss.detach().item()

        optimizer.zero_grad()
        our_loss.backward()
        optimizer.step()

        _update_progress_bar(cfg, model, test_dataloader, progress_bar, total_loss, idx, print_stats)

    return total_loss / len(train_dataloader)


def _update_progress_bar(cfg, model, test_dataloader, progress_bar, total_loss, idx, print_stats=False):
    """
    Utility to update the progress bar inside the training function
    Args:
        cfg (cfgNode): Model configurations
        model (torch.nn.model): Video model
        test_dataloader (DatasetLoader): testing dataset loader
        total_loss (float): Total loss so far in this training epoch
        idx (int): Index of batch inside the training epoch
        progress_bar (tqdm): if print_status, will be used to print a training progress bar
        print_stats (Bool): Whether to print stats or not
    """
    if print_stats:
        if 'auc' not in _update_progress_bar.__dict__:
            _update_progress_bar.auc = None

        if cfg.TRAIN.ENABLE_EVAL_BATCH and idx % cfg.TRAIN.EVAL_BATCH_PERIOD == 0:
            _update_progress_bar.auc, _, _, _ = test_engine.test(cfg, model, test_dataloader, False)
            model.train()

        progress_bar.update(n=1)

        if _update_progress_bar.auc is not None:
            progress_bar.set_postfix(loss=total_loss / (idx + 1), AUC=_update_progress_bar.auc)
        else:
            progress_bar.set_postfix(loss=total_loss / (idx + 1))
