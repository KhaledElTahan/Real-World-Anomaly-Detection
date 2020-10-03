"""Extract video features from the dataset using the backbone model."""
#from src.datasets import loader
from src.models import backbone_helper
from src.datasets import ucf_anomaly_detection

def extract(cfg):
    # backbone_model = backbone_helper.load_model(cfg)

    dataset = ucf_anomaly_detection.UCFAnomalyDetection(cfg, "test")
