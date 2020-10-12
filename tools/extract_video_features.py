"""Extract video features from the dataset using the backbone model."""

import torch
from tabulate import tabulate

from src.models import backbone_helper
from src.datasets import ucf_anomaly_detection
from src.datasets import utils
from src.utils import debugutils

@torch.no_grad()
def extract(cfg):

    temp_read_features = cfg.DATA.READ_FEATURES
    temo_extract_enabled = cfg.EXTRACT.ENABLE

    cfg.EXTRACT.ENABLE = True # Force train dataset to get items without respect to anomalies
    cfg.DATA.READ_FEATURES = False # Force read videos

    dataset_train = ucf_anomaly_detection.UCFAnomalyDetection(cfg, "train")
    dataset_test = ucf_anomaly_detection.UCFAnomalyDetection(cfg, "test")

    backbone_model = backbone_helper.load_model(cfg)
    backbone_model.eval()

    _print_extract_stats(cfg)

    for cur_iter, (frames, label, annotation) in enumerate(dataset_test):
        
        # First Index is used to distinguish between normal and anomaly video
        # Since we only use feature extraction, then all will be considered the same
        # Second Index is ued to distinguish between pathways
        frames_batches = utils.frames_to_batches_of_frames_batches(cfg, frames[0])
        
        for frames_batch in frames_batches:
            debugutils.print_tensors_nicely(frames_batch)
            preds = backbone_model(frames_batch)
            debugutils.print_tensors_nicely(preds)
            exit()


        exit()

    for cur_iter, (frames, label, annotation) in enumerate(test_loader):
        print(len(frames))
        print(frames[0][0].shape)
        print(frames[0][1].shape)
        #preds = backbone_model(frames[0])
        #print(preds.shape)


    cfg.DATA.READ_FEATURES = temp_read_features
    cfg.EXTRACT.ENABLE = temo_extract_enabled


def _print_extract_stats(cfg):
    """
    Prints a summary of the extraction process
    """
    print("Extraction Summary:")

    headers = ["Attribute", "Value"]
    table = [
        ["Frames inner batch size", cfg.EXTRACT.FRAMES_BATCH_SIZE],
        ["Frames stack batch size", cfg.EXTRACT.FRAMES_BATCHES_BATCH_SIZE],
    ]

    table = [x for x in table if x is not None]

    print(tabulate(table, headers, tablefmt="pretty", colalign=("center", "left")))
    print()