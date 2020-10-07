"""Extract video features from the dataset using the backbone model."""
#from src.datasets import loader
from src.models import backbone_helper
from src.datasets import ucf_anomaly_detection

def extract(cfg):

    temp_read_features = cfg.DATA.READ_FEATURES
    cfg.DATA.READ_FEATURES = False

    dataset_train = ucf_anomaly_detection.UCFAnomalyDetection(cfg, "train")
    print()
    dataset_test = ucf_anomaly_detection.UCFAnomalyDetection(cfg, "test")

    # backbone_model = backbone_helper.load_model(cfg)

    cfg.DATA.READ_FEATURES = temp_read_features