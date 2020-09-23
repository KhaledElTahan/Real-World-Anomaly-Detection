"""Extract video features from the dataset using the backbone model."""
#from src.datasets import loader
from src.models import backbone_helper

def extract(cfg):
    backbone_model = backbone_helper.load_model(cfg)

    print(backbone_model)
