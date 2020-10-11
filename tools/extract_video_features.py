"""Extract video features from the dataset using the backbone model."""
#from src.datasets import loader
from src.models import backbone_helper
from src.datasets import ucf_anomaly_detection
from src.datasets import utils

def extract(cfg):

    temp_read_features = cfg.DATA.READ_FEATURES
    temo_extract_enabled = cfg.EXTRACT.ENABLE

    cfg.EXTRACT.ENABLE = True # Force train dataset to get items without respect to anomalies
    cfg.DATA.READ_FEATURES = False # Force read videos

    dataset_train = ucf_anomaly_detection.UCFAnomalyDetection(cfg, "train")
    dataset_test = ucf_anomaly_detection.UCFAnomalyDetection(cfg, "test")

    frames, label, annotation = dataset_test[0]

    # First Index is used to distinguish between normal and anomaly video
    # Since we only use feature extraction, then all will be considered the same
    # Second Index is ued to distinguish between pathways
    slow_frames = frames[0][0]
    fast_frames = frames[0][1]

    print("Slow Frames: {}".format(slow_frames.shape))
    print("Fast Frames: {}".format(fast_frames.shape))

    frames, num_batches = utils.frames_to_frames_batches(cfg, frames[0])

    print("Slow Frames first batch: {}".format(frames[0][0].shape))
    print("Fast Frames first batch: {}".format(frames[1][0].shape))

    print("Number of batches: {}".format(num_batches))
    print("")

    # backbone_model = backbone_helper.load_model(cfg)

    cfg.DATA.READ_FEATURES = temp_read_features
    cfg.EXTRACT.ENABLE = temo_extract_enabled