"""Utility used to handle absoule paths"""
from pathlib import Path

def get_app_path():
    """Retrieve the absolute Path object of the app directory"""
    return Path(__file__).absolute().parents[2] 

def get_src_path():
    """Retrieve the absolute Path object of the src directory"""
    return Path(__file__).absolute().parents[1]

def get_models_path():
    """Retrieve the absolute Path object of the models directory"""
    return Path(__file__).absolute().parents[1] / "models"

def get_checkpoints_path():
    """Retrieve the absolute Path object of the checkpoints directory"""
    return get_app_path() / "checkpoints"

def get_datasets_path():
    """Retrieve the absolute Path object of the datasets directory"""
    return get_app_path() / "datasets"

def get_specific_dataset_path(cfg, features=False):
    """
    Retrieve the absolute Path object of a specific dataset directory
    Args:
        cfg (CfgNode): configs.
        mode (string): Options includes `train`, `val`, or `test` mode.
        features (Bool):
            if true, retrieve the features dataset directory
            if false, retrieve the videos dataset directory
    """
    dataset_directory = get_datasets_path() / cfg.DATA.PATH_TO_DATA_DIR

    if features: # Read features
        features_name = cfg.BACKBONE.NAME + "_" + cfg.TRANSFORM.CODE + "_" + \
            str(cfg.EXTRACT.FRAMES_BATCH_SIZE) + "x" + str(cfg.EXTRACT.NUMBER_OUTPUT_SEGMENTS)
        dataset_directory = dataset_directory / "features" / features_name
    else: # Read videos
        dataset_directory = dataset_directory / "videos"

    return dataset_directory

def get_configs_path():
    """Retrieve the absolute Path object of the configurations directory"""
    return get_app_path() / "configs"
    