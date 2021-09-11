"""Utils for PyTorch Models"""

import copy
import torch

from src.utils import funcutils
from src.models import backbone_helper


def create_state_dictionary(cfg, model):
    """
    Creates a cpu located model state dictionary without affecting the model
    Args:
        cfg (cfgNode): The video model configurations
        model (torch.nn.Module): the model
    Returns:
        model_state_dictionary (dict): The model state dictionary of model
    """
    model = copy.deepcopy(model)

    if cfg.NUM_GPUS > 1:
        model = model.module

    model = model.cpu()

    return model.state_dict()


@funcutils.force_garbage_collection(before=True, after=True)
@torch.no_grad()
def get_features_length(cfg, backbone_model):
    """
    Feed the backbone with least dummy input with the same dataset shape
    and find the length of the features
    Args:
        cfg (cfgNode): The video model configurations
        backbone_model (torch.nn.Module): The feature extractor model
    Returns:
        (int): The length of the features dimension
    """
    batch_size = 1
    channels = cfg.DATA.INPUT_CHANNEL_NUM
    frames_inner_batch = cfg.EXTRACT.FRAMES_BATCH_SIZE
    height = cfg.DATA.SCALES[0]
    width = cfg.DATA.SCALES[1]

    # First, retrieve the backbone configurations file
    backbone_cfg = backbone_helper.get_backbone_merged_cfg(cfg)

    if backbone_cfg.MODEL.ARCH in backbone_cfg.MODEL.SINGLE_PATHWAY_ARCH:
        inputs = [torch.rand(size = (batch_size, channels, frames_inner_batch, height, width))]
    elif backbone_cfg.MODEL.ARCH in backbone_cfg.MODEL.MULTI_PATHWAY_ARCH:
        alpha = backbone_cfg.SLOWFAST.ALPHA
        inputs = [
                torch.rand(size = (batch_size, channels, frames_inner_batch//alpha, height, width)),
                torch.rand(size = (batch_size, channels, frames_inner_batch, height, width))
            ]
    else:
        raise NotImplementedError(
            "Model arch {} is not in {}".format(
                backbone_cfg.MODEL.ARCH,
                backbone_cfg.MODEL.SINGLE_PATHWAY_ARCH + backbone_cfg.MODEL.MULTI_PATHWAY_ARCH,
            )
        )

    if cfg.NUM_GPUS > 0:
        for idx, _ in enumerate(inputs):
            inputs[idx] = inputs[idx].cuda()

    backbone_model.eval()
    _, feats = backbone_model(inputs)

    feats = feats.detach()

    if cfg.NUM_GPUS > 0:
        for idx, _ in enumerate(inputs):
            inputs[idx] = inputs[idx].cpu()
        feats = feats.cpu()

    return feats.shape[1]
