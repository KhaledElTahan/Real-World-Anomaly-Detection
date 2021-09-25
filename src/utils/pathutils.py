"""Utility used to handle absoule paths"""
from pathlib import Path
from src.utils import infoutils

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


def get_models_checkpoint_directory_path(cfg):
    """
    Retrieve the absolute Path object of the directory that has all
        checkpoints of the whole model
    Args
        cfg (CfgNode): Model Configurations
    Returns
        dir_path (Path): Path of the directory
    """
    return get_checkpoints_path() / cfg.MODEL.CHECKPOINTS_DIRECTORY


def get_model_checkpoint_path(cfg):
    """
    Retrieve the absolute Path object of the checkpoint of the whole model
    Args
        cfg (CfgNode): Model Configurations
    Returns
        cp_path (Path): Path of the checkpoint
    """
    cp_path = get_models_checkpoint_directory_path(cfg)
    cp_path = cp_path / (infoutils.get_full_model_name(cfg) + '.pt')

    return cp_path


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
        dataset_directory = dataset_directory / "features" / \
            infoutils.get_dataset_features_name(cfg)
    else: # Read videos
        dataset_directory = dataset_directory / "videos"

    return dataset_directory

def get_configs_path():
    """Retrieve the absolute Path object of the configurations directory"""
    return get_app_path() / "configs"


def change_extension(video_name, old_ext, new_ext):
    """
    Changes the file name's extension
    Args:
        old_ext (String): The old extension of the file
        new_ext (String): The new extension of the file
    Examples:
        change_extension("video.mp4 label 1 1", "mp4", "rar") -> "video.rar label 1 1"
    """
    return video_name.replace(old_ext, new_ext)


def video_path_to_features_path(cfg, video_path :Path):
    """
    Convert video path to its features file path
    Args:
        cfg (cfgNode): Video model configurations node
        video_path (pathlib.Path): Path of the video
    Returns:
        features_path (pathlib.Path): Path of the features file
    """
    parent_directory = get_specific_dataset_path(cfg, features=True) / video_path.parent.name
    return parent_directory / change_extension(str(video_path.name), cfg.DATA.EXT, cfg.EXTRACT.FEATURES_EXT)


def features_path_to_video_path(cfg, features_path :Path):
    """
    Convert video path to its features file path
    Args:
        cfg (cfgNode): Video model configurations node
        features_path (pathlib.Path): Path of the features file
    Returns:
        video_path (pathlib.Path): Path of the video
    """
    parent_directory = get_specific_dataset_path(cfg, features=False) / features_path.parent.name
    return parent_directory / change_extension(str(features_path.name), cfg.EXTRACT.FEATURES_EXT, cfg.DATA.EXT)
