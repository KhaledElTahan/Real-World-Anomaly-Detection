"""Model training engine"""

from src.models import losses
from src.utils import funcutils
import torch
from tqdm import tqdm

from src.engine import test_engine


@torch.enable_grad()
def train(cfg, model, optimizer, train_dataloader, test_dataloader, current_epoch, print_stats=False):
    """
    Chooses the training policy then train the model one epoch on the data loader
    Args:
        cfg (cfgNode): Model configurations
        model (torch.nn.model): Video model
        optimizer (torch.nn.optimizer): The used optimizer
        train_dataloader (DatasetLoader): training dataset loader
        test_dataloader (DatasetLoader): testing dataset loader
        current_epoch (int): Current epoch for the training process
        print_stats (Bool): Whether to print stats or not
    Returns:
        loss_value (float): The loss of one epoch
    """
    assert cfg.TRAIN.TYPE in ['MIL', 'PL', 'PL-MIL']

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
        loss_value = generic_train( 
            cfg,
            model,
            losses.get_loss_class(cfg, False),
            optimizer,
            train_dataloader,
            test_dataloader, 
            print_stats,
            progress_bar
        )
    elif cfg.TRAIN.TYPE == "PL":
        loss_value = generic_train( 
            cfg,
            model,
            losses.get_loss_class(cfg, True),
            optimizer,
            train_dataloader,
            test_dataloader,
            print_stats,
            progress_bar
        )
    elif cfg.TRAIN.TYPE == "PL-MIL":
        loss_value = pseudo_labels_MIL_train(
            cfg, model, optimizer, train_dataloader, test_dataloader, print_stats, progress_bar
        )


    if print_stats:
        progress_bar.close()

    return loss_value


@funcutils.profile(apply=False, lines_to_print=15, strip_dirs=True)
def generic_train(cfg, model, loss_class, optimizer, train_dataloader, test_dataloader, print_stats=False, progress_bar=None):
    """
    Generic training method that runs any training mechanism based on the loss_class.
    Train the model one epoch on the data loader.
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

    for idx, batches in enumerate(train_dataloader):
        if cfg.NUM_GPUS > 0:
            features = [batch["features_batched"].cuda() for batch in batches]
        else:
            features = [batch["features_batched"] for batch in batches]

        preds = [model(features_batch) for features_batch in features]

        loss = loss_class(cfg, preds)
        our_loss = loss()

        # Make sure testing inside training doesn't cause problems
        assert our_loss.requires_grad

        total_loss += our_loss.detach().item()

        optimizer.zero_grad()
        our_loss.backward()
        optimizer.step()

        _update_progress_bar(
            cfg, model, test_dataloader, progress_bar, total_loss, idx, print_stats
        )

    return total_loss / len(train_dataloader)


@funcutils.profile(apply=False, lines_to_print=15, strip_dirs=True)
def pseudo_labels_MIL_train(cfg, model, optimizer, train_dataloader, test_dataloader, print_stats=False, progress_bar=None):
    """
    Mltiple instance learning + pseudo labels training
    Train the model one epoch on the data loader
    Args:
        cfg (cfgNode): Model configurations
        model (torch.nn.model): Video model
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

    for idx, (org_normal_batch, org_anomaleous_batch, aug_normal_batch, aug_anomaleous_batch) \
            in enumerate(train_dataloader):
        features_org_normal_batch = org_normal_batch["features_batched"]
        features_org_anomaly_batch = org_anomaleous_batch["features_batched"]

        if cfg.NUM_GPUS > 0:
            features_org_normal_batch = features_org_normal_batch.cuda()
            features_org_anomaly_batch = features_org_anomaly_batch.cuda()

        normal_preds = model(features_org_normal_batch)
        anomaly_preds = model(features_org_anomaly_batch)

        loss = loss_class(normal_preds, anomaly_preds)
        our_loss = loss()

        assert our_loss.requires_grad # to make sure testing inside training doesn't cause problems

        total_loss += our_loss.detach().item()

        optimizer.zero_grad()
        our_loss.backward()
        optimizer.step()

        _update_progress_bar(
            cfg, model, test_dataloader, progress_bar, total_loss, idx, print_stats
        )

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
