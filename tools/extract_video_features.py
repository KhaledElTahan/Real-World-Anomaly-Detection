"""Extract video features from the dataset using the backbone model."""

import torch
from tabulate import tabulate

from src.models import backbone_helper
from src.datasets import ucf_anomaly_detection
from src.datasets import utils
from src.utils import debugutils
from src.utils import modelutils

@torch.no_grad()
def extract(cfg):
    """
    Main tool for feature extraction
    Args:
        cfg (cfgNode): Video Model Configurations
    """
    simple_test()
    temp_read_features = cfg.DATA.READ_FEATURES
    temo_extract_enabled = cfg.EXTRACT.ENABLE

    cfg.EXTRACT.ENABLE = True # Force train dataset to get items without respect to anomalies
    cfg.DATA.READ_FEATURES = False # Force read videos

    dataset_train = ucf_anomaly_detection.UCFAnomalyDetection(cfg, "train")
    dataset_test = ucf_anomaly_detection.UCFAnomalyDetection(cfg, "test")

    backbone_model = backbone_helper.load_model(cfg)
    backbone_model.eval()

    features_length = modelutils.get_features_length(cfg, backbone_model)

    _print_extract_stats(cfg, features_length)

    if features_length != cfg.BACKBONE.FEATURES_LENGTH:
        print("Warning: Set cfg.BACKBONE.FEATURES_LENGTH with value {}", features_length)

    for cur_iter, (frames, label, annotation) in enumerate(dataset_test):
   
        # First Index is used to distinguish between normal and anomaly video
        # Since we only use feature extraction, then all will be considered the same
        # Second Index is ued to distinguish between pathways
        frames_batches = utils.frames_to_batches_of_frames_batches(cfg, frames[0])

        for frames_batch in frames_batches:
            debugutils.print_tensors_nicely(frames_batch)
            preds, features = backbone_model(frames_batch)

            debugutils.print_tensors_nicely(preds)
            debugutils.print_tensors_nicely(features)
            exit()


        exit()

    cfg.DATA.READ_FEATURES = temp_read_features
    cfg.EXTRACT.ENABLE = temo_extract_enabled


def _print_extract_stats(cfg, features_length):
    """
    Prints a summary of the extraction process
    Args:
        cfg (cfgNode): Video model configurations
        features_length (Int): Length of extracted features dimension
    """
    print("Extraction Summary:")

    headers = ["Attribute", "Value"]
    table = [
        ["Frames inner batch size", cfg.EXTRACT.FRAMES_BATCH_SIZE],
        ["Frames stack batch size", cfg.EXTRACT.FRAMES_BATCHES_BATCH_SIZE],
        ["Machine Type", "CPU" if cfg.NUM_GPUS == 0 else "GPU"],
        ["No. GPUs", cfg.NUM_GPUS],
        ["CFG. Features Length", cfg.BACKBONE.FEATURES_LENGTH],
        ["Actual Features Length", features_length],
    ]

    table = [x for x in table if x is not None]

    print(tabulate(table, headers, tablefmt="pretty", colalign=("center", "left")))
    print()


def simple_test():
    utils.changes_segments_number(torch.rand(size = (2, 3, 4)), 8)


    exit()