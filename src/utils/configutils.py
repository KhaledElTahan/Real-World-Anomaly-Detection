"""Utilities related to configuration nodes"""
import operator
from fvcore.common.config import CfgNode


def set_cfg_val(cfg, attrib, attrib_val):
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

        set_cfg_val(inner_cfg_node, attrib.replace(first_attrib + '.', '', 1), attrib_val)
        setattr(cfg, first_attrib, inner_cfg_node)
    else:
        setattr(cfg, attrib, attrib_val)


def unify_config_attributes(src_cfg, dst_cfg, attributes_list):
    """
    Merge the configurations from src_cfg into dst_cfg based on attributes_list
    Args:
        src_cfg (cfgNode): The source configuration node
        dst_cfg (cfgNode): The destination configuration node
        attributes_list (List) : attributes list to be copied from source cfg
            to destination cfg
    Example:
        attributes_list = [
            "NUM_GPUS",
            "NUM_SHARDS",
            "MODEL.ARCH",
        ]
    """
    for attrib in attributes_list:
        attrib_val = operator.attrgetter(attrib)(src_cfg)
        set_cfg_val(dst_cfg, attrib, attrib_val)

    return dst_cfg