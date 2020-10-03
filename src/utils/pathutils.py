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

def get_configs_path():
    """Retrieve the absolute Path object of the configurations directory"""
    return get_app_path() / "configs"