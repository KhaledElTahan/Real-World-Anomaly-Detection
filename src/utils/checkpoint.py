"""Checkpoint utilities"""

import torch

from fvcore.common.config import CfgNode
from src.utils import pathutils, modelutils


def save_checkpoint(cfg, optimizer, model, epoch_auc, best_model_state_dict, best_auc, best_epoch):
    """
    Saves a checkpoint of the model and the training details,
    Always saves a checkpoint with completed epochs, and if is_best
    is True it saves another checkpoint to be marked as best
    Specifically, it saves:
        1) Model configurations, used for assertion on loading
        2) Optimizer state dictionary
        3) Model state dictionary of current epoch
        4) Area under the ROC of current epoch
        5) Number of completed epochs
        6) Model state dictionary of best AUC
        7) Best area under the ROC
        8) Number of completed epochs for best AUC
    Args:
        cfg (CfgNode): Model Configurations
        Model (torch.nn.Module): The video model
        optimizer (torch.nn.optimizer): The optimizer
        auc (float): Area under the ROC curve of the model
        is_best (bool): Is this the model that achieved best auc so far?
    """
    model_state_dict = modelutils.create_state_dictionary(cfg, model)
    cp_path = pathutils.get_model_checkpoint_path(cfg)

    torch.save(
        {
            "cfg": cfg.dump(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_state_dict": model_state_dict,
            "auc": epoch_auc,
            "epoch": cfg.TRAIN.CURRENT_EPOCH,
            "best_model_state_dict": best_model_state_dict,
            "best_auc": best_auc,
            "best_epoch": best_epoch,
        }, cp_path
    )


def _assert_checkpoint(cfg, checkpoint_cfg):
    """
    Asserts that the saved checkpoint's configurations is
    compatible with current configurations
    """
    assert cfg.MODEL.MODEL_NAME == checkpoint_cfg.MODEL.MODEL_NAME
    assert cfg.MODEL.LOSS_FUNC == checkpoint_cfg.MODEL.LOSS_FUNC
    assert cfg.MODEL.SIGN == checkpoint_cfg.MODEL.SIGN
    assert cfg.TRAIN.TYPE == checkpoint_cfg.TRAIN.TYPE
    assert cfg.EXTRACT.FRAMES_BATCH_SIZE == checkpoint_cfg.EXTRACT.FRAMES_BATCH_SIZE
    assert cfg.EXTRACT.NUMBER_OUTPUT_SEGMENTS == checkpoint_cfg.EXTRACT.NUMBER_OUTPUT_SEGMENTS
    assert cfg.BACKBONE.NAME == checkpoint_cfg.BACKBONE.NAME
    assert cfg.TRANSFORM.CODE == checkpoint_cfg.TRANSFORM.CODE
    assert cfg.OPTIMIZER.NAME == checkpoint_cfg.OPTIMIZER.NAME
    assert cfg.OPTIMIZER.BASE_LR == checkpoint_cfg.OPTIMIZER.BASE_LR
    assert cfg.OPTIMIZER.WEIGHT_DECAY == checkpoint_cfg.OPTIMIZER.WEIGHT_DECAY
    assert cfg.OPTIMIZER.EPS == checkpoint_cfg.OPTIMIZER.EPS


def load_checkpoint(cfg, assert_cfg=True, checkpoint_path=None):
    """
    Loads a checkpoint
    Args
        cfg (cfgNode): Model configurations
        assert_cfg (Bool): if True, will assert that the loaded checkpoint matches
            the current configurations
        checkpoint_path (Path): If None, default checkpoint will be loaded by the guide
            of the cfg, otherwise the requested path will be loaded
    Returns:
        checkpoint_cfg (cfgNide): Checkpoint configurations
        optimizer_state_dict: Optimizer state dictionary
        model_state_dict: Model state dictionary of current epoch
        auc: Area under the ROC of current epoch
        epoch: Number of completed epochs
        best_model_state_dict: Model state dictionary of best AUC
        best_auc: Best area under the ROC
        best_epoch: Number of completed epochs for best AUC
    """
    if checkpoint_path is None:
        cp_path = pathutils.get_model_checkpoint_path(cfg)
    else:
        cp_path = checkpoint_path

    assert cp_path.exists()

    # Load the checkpoint on CPU to avoid GPU mem spike.
    loaded = torch.load(cp_path, map_location='cpu')

    checkpoint_cfg = CfgNode.load_cfg(loaded['cfg'])
    if assert_cfg:
        _assert_checkpoint(cfg, checkpoint_cfg)

    optimizer_state_dict = loaded['optimizer_state_dict']
    model_state_dict = loaded['model_state_dict']
    auc = loaded['auc']
    epoch = loaded['epoch']
    best_model_state_dict = loaded['best_model_state_dict']
    best_auc = loaded['best_auc']
    best_epoch = loaded['best_epoch']

    return checkpoint_cfg, optimizer_state_dict, model_state_dict, \
        auc, epoch, best_model_state_dict, best_auc, best_epoch


def checkpoint_exists(cfg):
    """
    Check whether there's a saved checkpoint
    Args:
        cfg (cfgNode): Model configurations
    Returns:
        exits (Bool): Whether a checkpoint exists or not
    """
    return pathutils.get_model_checkpoint_path(cfg).exists()
