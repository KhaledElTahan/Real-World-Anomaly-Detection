"""Extract video features from the dataset using the backbone model."""
import torch

from src.models import backbone_helper
from src.datasets import ucf_anomaly_detection
from src.datasets import utils
from src.datasets import loader

@torch.no_grad()
def extract(cfg):

    temp_read_features = cfg.DATA.READ_FEATURES
    temo_extract_enabled = cfg.EXTRACT.ENABLE

    cfg.EXTRACT.ENABLE = True # Force train dataset to get items without respect to anomalies
    cfg.DATA.READ_FEATURES = False # Force read videos

    dataset_train = ucf_anomaly_detection.UCFAnomalyDetection(cfg, "train")
    dataset_test = ucf_anomaly_detection.UCFAnomalyDetection(cfg, "test")

    frames, label, annotation = dataset_test[1]

    # First Index is used to distinguish between normal and anomaly video
    # Since we only use feature extraction, then all will be considered the same
    # Second Index is ued to distinguish between pathways
    slow_frames = frames[0][0]
    fast_frames = frames[0][1]

    print("Slow Frames: {}".format(slow_frames.shape))
    print("Fast Frames: {}".format(fast_frames.shape))

    frames, num_batches = utils.frames_to_frames_batches(cfg, frames[0], dim=2)

    print("Slow Frames first batch: {}".format(frames[0][0].shape))
    print("Fast Frames first batch: {}".format(frames[1][0].shape))

    print("len(SlowFrames): {}".format(len(frames[0])))
    print("len(FastFrames): {}".format(len(frames[1])))

    print("Number of batches: {}".format(num_batches))
    print("")

    test_loader = loader.construct_loader(cfg, "test")
    train_loader = loader.construct_loader(cfg, "train")

    backbone_model = backbone_helper.load_model(cfg)
    backbone_model.eval()

    for cur_iter, (frames, label, annotation) in enumerate(test_loader):
        print(frames[0][0].shape)
        preds = backbone_model(frames[0])
        print(preds.shape)


    cfg.DATA.READ_FEATURES = temp_read_features
    cfg.EXTRACT.ENABLE = temo_extract_enabled