"""Add custom configs and default values"""


def add_custom_untracked_config(cfg):
    """
    First Highest priority configurations, untracked on git
    NOTE: You must run the following git command in the project directory to be effective
        git update-index --assume-unchanged src/config/custom_untracked_config.py
    NOTE: To undo the above command and make changes noticeable by git
        (to make changes for the config functionalty and not the actual configs)
        git update-index --no-assume-unchanged src/config/custom_untracked_config.py
    Args
        cfg (cfgNode): Model configurations
    """
