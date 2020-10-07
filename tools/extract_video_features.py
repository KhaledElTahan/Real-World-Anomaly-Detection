"""Extract video features from the dataset using the backbone model."""
#from src.datasets import loader
from src.models import backbone_helper
from src.datasets import ucf_anomaly_detection

def extract(cfg):

    temp_read_features = cfg.DATA.READ_FEATURES
    cfg.DATA.READ_FEATURES = False

    dataset_train = ucf_anomaly_detection.UCFAnomalyDetection(cfg, "train")
    dataset_test = ucf_anomaly_detection.UCFAnomalyDetection(cfg, "test")

    frames, labels, annotations = dataset_test[0]

    fast_frames = frames[0][0]
    slow_frames = frames[0][1]

    print("Fast Frames: {}".format(fast_frames.shape))
    print("Slow Frames: {}".format(slow_frames.shape))

    # backbone_model = backbone_helper.load_model(cfg)

    cfg.DATA.READ_FEATURES = temp_read_features