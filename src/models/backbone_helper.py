"""Helper to reuse the backbone model"""
import torch
import operator
from fvcore.common.config import CfgNode
import src.utils.pathutils as pathutils
from src.models.slowfast.config.defaults import get_cfg as get_backbone_default_cfg
from src.models.slowfast.models import build_model
import src.models.slowfast.utils.checkpoint as cu


def _set_cfg_val(cfg, attrib, attrib_val):
    """
    Sets value of configuration attribute recursivly,
    If attribute exists, then its value is changed, if it doesn't exist,
    The attribute is created recursivly then set to attrib_val
    Args:
        cfg (cfgNode): Any configuration node
        attrib (String): Attribute name
        attrib_val: Attribute value
    Example:
        attrib: cfg.TRAIN.GPUS.NUMBER
        attrib_value: 15
    """
    if '.' in attrib:
        first_attrib = attrib.split('.')[0]

        if hasattr(cfg, first_attrib):
            inner_cfg_node = getattr(cfg, first_attrib)
        else:
            inner_cfg_node = CfgNode()

        _set_cfg_val(inner_cfg_node, attrib.replace(first_attrib + '.', '', 1), attrib_val)
        setattr(cfg, first_attrib, inner_cfg_node)
    else:
        setattr(cfg, attrib, attrib_val)


def _merge_configurations(backbone_cfg, cfg):
    """
    Merge the configurations from cfg into backbone_cfg based on list
    of attributes names stored in cfg.BACKBONE.MERGE_CFG_LIST
    Args:
        backbone_cfg (cfgNode): The backbone model configuration file
        cfg (cfgNode): The video model configuration file
    Example:
        cfg.BACKBONE.MERGE_CFG_LIST = [
            "NUM_GPUS",
            "NUM_SHARDS",
            "MODEL.ARCH",
        ]
    """
    for attrib in cfg.BACKBONE.MERGE_CFG_LIST:
        attrib_val = operator.attrgetter(attrib)(cfg)
        _set_cfg_val(backbone_cfg, attrib, attrib_val)

    return backbone_cfg


def _load_native_backbone_cfg(cfg):
    """
    Loads the configuration file of the backbone
    Args:
        cfg: Uses the cfg.BACKBONE.CONFIG_FILE_PATH to retrieve the
            backbone configuration file
    """

    backbone_cfg = get_backbone_default_cfg()
    backbone_cfg.merge_from_file(pathutils.get_configs_path() / cfg.BACKBONE.CONFIG_FILE_PATH)

    return backbone_cfg


def get_backbone_merged_cfg(cfg):
    """
    Loads the configuration file of the backbone, merges it with cfg
        then return the merged backbone configuration
    Args:
        cfg: The video model configuration file
    """
    return _merge_configurations(_load_native_backbone_cfg(cfg), cfg)


def load_model(cfg):
    """
    Create a backbone model from the configurations and load its weights
    Args:
        cfg: The video model configuration file
    """

    backbone_cfg = get_backbone_merged_cfg(cfg)
    backbone_model = build_model(backbone_cfg)

    cu.load_checkpoint(
        pathutils.get_checkpoints_path() / cfg.BACKBONE.CHECKPOINT_FILE_PATH,
        backbone_model,
        cfg.NUM_GPUS > 1,
        None,
        inflation=False,
        convert_from_caffe2=backbone_cfg.TRAIN.CHECKPOINT_TYPE == "caffe2"
    )

    return backbone_model
