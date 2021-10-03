"""Model training engine"""

import torch
from tqdm import tqdm

from src.models import losses
from src.engine import test_engine
from src.utils import funcutils


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
        progress_bar = _get_progress_bar(cfg, train_dataloader)

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

        auc = _evaluate_per_batch(cfg, model, test_dataloader, idx)
        _update_progress_bar(progress_bar, total_loss, idx, auc, print_stats, **loss.get_progress_bar_info())

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
        print_stats (Bool): Whether to print stats or not
        progress_bar (tqdm): if print_status, will be used to print a training progress bar
    Retunrs:
        loss_value (float): mean loss per example in this training epoch
    """

    training_selection = _get_training_selection_for_epoch_PL_MIL(cfg)

    if training_selection == "MIL":
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
    elif training_selection == "PL":
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

    return loss_value


def _get_training_selection_for_epoch_PL_MIL(cfg):
    """
    Utility used to get the training selection per epoch for PL-MIL Training
    Args:
        cfg (cfgNode): Model configurations
    Returns:
        training_selection
    """
    assert cfg.TRAIN.TYPE == 'PL-MIL'
    assert len(cfg.TRAIN.PL_MIL_INTERVALS) > 0
    assert len(cfg.TRAIN.PL_MIL_PERCENTAGES) == len(cfg.TRAIN.PL_MIL_INTERVALS)

    sum_intervals = 0
    interval_index = len(cfg.TRAIN.PL_MIL_PERCENTAGES) - 1
    remaining_intervals = cfg.TRAIN.CURRENT_EPOCH

    for index, interval in enumerate(cfg.TRAIN.PL_MIL_INTERVALS):
        sum_intervals += interval

        if cfg.TRAIN.CURRENT_EPOCH <= sum_intervals:
            interval_index = index
            break

        remaining_intervals -= interval

    # Consider after last interval scenario
    if remaining_intervals > cfg.TRAIN.PL_MIL_INTERVALS[interval_index]:
        remaining_intervals %= cfg.TRAIN.PL_MIL_INTERVALS[interval_index]
    
    if remaining_intervals == 0:
        remaining_intervals = cfg.TRAIN.PL_MIL_INTERVALS[interval_index]
    
    greater = False
    if remaining_intervals > \
        cfg.TRAIN.PL_MIL_INTERVALS[interval_index] * cfg.TRAIN.PL_MIL_PERCENTAGES[interval_index]:
        greater = True

    #     Table      | MIL_FIRST (True) | MIL_First (False)
    # Greater (False)|       'MIL'      |       'PL'
    # Greater (True) |       'PL'       |       'MIL'
    if greater ^ cfg.TRAIN.PL_MIL_MILFIRST:
        training_selection = 'MIL'
    else:
        training_selection = 'PL'

    return training_selection


def _evaluate_per_batch(cfg, model, test_dataloader, idx):
    """
    Evalues the model inside the epoch
    Args:
        cfg (cfgNode): Model configurations
        model (torch.nn.model): Video model
        test_dataloader (DatasetLoader): testing dataset loader
        idx (int): Index of batch inside the training epoch
    Returns:
        auc (float | None): Area under the ROC curve, or None if no evaluation is needed
    """
    auc = None
    if cfg.TRAIN.ENABLE_EVAL_BATCH and idx % cfg.TRAIN.EVAL_BATCH_PERIOD == 0:
        auc, _, _, _ = test_engine.test(cfg, model, test_dataloader, False)
        model.train()

    return auc


def _get_progress_bar(cfg, train_dataloader):
    """
    Utility to create the progress with regards to cfg
    Args:
        cfg (cfgNode): Model configurations
        train_dataloader (DatasetLoader): training dataset loader
    Returns:
        progress_bar (tqdm): if print_status, will be used to print a training progress bar
    """

    assert cfg.TRAIN.TYPE in ['MIL', 'PL', 'PL-MIL']

    if cfg.TRAIN.TYPE == "MIL":
        description = "MIL Train Progress - Epoch (" + str(cfg.TRAIN.CURRENT_EPOCH) + ")"
        colour = "green"
    elif cfg.TRAIN.TYPE == "PL":
        description = "PL Train Progress - Epoch (" + str(cfg.TRAIN.CURRENT_EPOCH) + ")"
        colour = "blue"
    elif cfg.TRAIN.TYPE == "PL-MIL":
        training_selection = _get_training_selection_for_epoch_PL_MIL(cfg)
        description = "MIL-PL: (" + training_selection + ") Train Progress - Epoch (" +\
            str(cfg.TRAIN.CURRENT_EPOCH) + ")"
        if training_selection == "MIL":
            colour = "green"
        else:
            colour = "blue"

    progress_bar = tqdm(
        total=len(train_dataloader),
        desc=description,
        unit="batch",
        colour=colour
    )

    return progress_bar


def _update_progress_bar(progress_bar, total_loss, idx, auc, print_stats, **kwargs):
    """
    Utility to update the progress bar inside the training function
    Args:
        progress_bar (tqdm): if print_status, will be used to print a training progress bar
        total_loss (float): Total loss so far in this training epoch
        idx (int): Index of batch inside the training epoch
        auc (float | None): Area under the ROC curve, or None if no evaluation is needed
        print_stats (Bool): Whether to print stats or not
    """
    if print_stats:
        if 'auc' not in _update_progress_bar.__dict__:
            _update_progress_bar.auc = None

        if auc is not None:
            _update_progress_bar.auc = auc

        progress_bar.update(n=1)

        if _update_progress_bar.auc is not None:
            progress_bar.set_postfix(loss=total_loss / (idx + 1), \
                AUC=_update_progress_bar.auc, **kwargs)
        else:
            progress_bar.set_postfix(loss=total_loss / (idx + 1), **kwargs)
