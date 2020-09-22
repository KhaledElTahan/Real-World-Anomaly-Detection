"""Extract video features from the dataset using the backbone model."""
import sys
import os
sys.path.insert(0, os.path.abspath('..'))

#from src.datasets import loader
from src.models import backbone_helper

def extract(cfg):
    backbone_helper.load_model(cfg)
