"""Checkpoint utilities"""

import torch
from src.models.build import build_model
from src.utils import pathutils


def save_checkpoint(cfg, model, optimizer, auc, is_best):
    """
    Saves a checkpoint of the model and the training details,
    Always saves a checkpoint with completed epochs, and if is_best
    is True it saves another checkpoint to be marked as best
    Specifically, it saves:
        1) Model state dictionary
        2) Optimizer state dictionary
        3) Number of completed epochs
        4) Number of GPUs model was trained on
        5) Area under the ROC Curve
    Args:
        cfg (CfgNode): Model Configurations
        Model (torch.nn.Module): The video model
        optimizer (torch.nn.optimizer): The optimizer
        auc (float): Area under the ROC curve of the model
        is_best (bool): Is this the model that achieved best auc so far?
    """
    def _save_checkpoint(cfg, model, optimizer, auc, cp_path):
        """Basic functionality for save_checkpoint"""

        if cfg.NUM_GPUS > 1:
            model_state_dict = model.module.state_dict()
        else:
            model_state_dict = model.state_dict()

        torch.save(
            {
                "model_state_dict": model_state_dict,
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": cfg.TRAIN.CURRENT_EPOCH,
                "gpus": cfg.NUM_GPUS,
                "auc": auc,
            }, cp_path
        )

    if is_best:
        cp_path = pathutils.get_model_checkpoint_path(cfg, True, None)
        _save_checkpoint(cfg, model, optimizer, auc, cp_path)

    cp_path = pathutils.get_model_checkpoint_path(cfg, False, cfg.TRAIN.EPOCH)
    _save_checkpoint(cfg, model, optimizer, auc, cp_path)





def load_checkpoint(cfg, is_best=True, epoch=None):
    """
    """
    cp_path = pathutils.get_model_checkpoint_path(cfg, False, cfg.TRAIN.EPOCH)

    loaded = torch.load(features_path)

    features_segments = loaded['features_segments']
    is_anomaly_segment = loaded['is_anomaly_segment']

    return features_segments, is_anomaly_segment

def _load_model_device(model, saved_device, current_device):
    pass