"""Helper to reuse the backbone model"""
import src.utils.pathutils as pathutils
from src.utils import configutils
from src.models.slowfast.config.defaults import get_cfg as get_backbone_default_cfg
from src.models.slowfast.models import build_model
import src.models.slowfast.utils.checkpoint as cu


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
    return configutils.unify_config_attributes(
                cfg,
                _load_native_backbone_cfg(cfg),
                cfg.BACKBONE.MERGE_CFG_LIST
            )


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
