"""Add git tracked custom configs and default values"""


def add_custom_tracked_config(cfg):
    """
    Second Highest priority configurations, tracked on git
    Args
        cfg (cfgNode): Model configurations
    """

    cfg.DATA.MAX_VIDEO_SIZE = 200 * 1024 * 1024
