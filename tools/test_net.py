"""Extract video features from the dataset using the backbone model."""

import gc
import time
import torch
from tabulate import tabulate
from tqdm import tqdm

from src.models import backbone_helper
from src.datasets import utils
from src.datasets import build
from src.utils import modelutils
from src.utils import infoutils
from src.utils import pathutils


def test(cfg):
    """
    Main tool for overall model teating
    Args:
        cfg (cfgNode): Video Model Configurations
    """
    temp_read_features = cfg.DATA.READ_FEATURES
    temp_extract_enabled = cfg.EXTRACT.ENABLE
    temp_backbone_trainable = cfg.BACKBONE.TRAINABLE


    cfg.EXTRACT.ENABLE = True # Force train dataset to get items without respect to anomalies
    cfg.DATA.READ_FEATURES = True # Force read videos
    cfg.BACKBONE.TRAINABLE = False # Force backbone to detach features

    datasets = []
    for split in cfg.EXTRACT.DATASET_SPLITS:
        datasets.append(build.build_dataset(cfg.EXTRACT.DATASET, cfg, split))

    full_model = None ## Load Model
    full_model.train() ## Model.train


    total_len = sum([len(dataset) for dataset in datasets])

    _print_test_stats(cfg)

    progress_bar = tqdm(total=total_len, desc="Feature Extraction Progress")
    for dataset in datasets:
        for _, (frames, _, annotation, video_index) in enumerate(dataset):

            features_path = pathutils.video_path_to_features_path(
                cfg, dataset.get_file_path(video_index)
            )
            if not cfg.EXTRACT.FORCE_REWRITE and features_path.exists():
                progress_bar.update(n=1)
                time.sleep(0.05)
                continue

            frames_batches = utils.frames_to_batches_of_frames_batches(cfg, frames[0])
            del frames

            features_batches = []
            for frames_batch in frames_batches:

                if cfg.NUM_GPUS > 0:
                    for i, _ in enumerate(frames_batch):
                        frames_batch[i] = frames_batch[i].cuda()

                _, features = backbone_model(frames_batch)

                if cfg.NUM_GPUS > 0:
                    for i, _ in enumerate(frames_batch):
                        frames_batch[i] = frames_batch[i].cpu()

                features_batches.append(
                    features.cpu() if cfg.NUM_GPUS > 0 else features
                )
            del frames_batches

            for i, _ in enumerate(features_batches):
                features_batches[i] = features_batches[i].detach()

            new_segments, is_anomaly_segment = utils.segmentize_features(
                cfg, torch.cat(features_batches), annotation[0]
            )
            del features_batches

            features_path.parent.mkdir(parents=True, exist_ok=True)

            torch.save(
                {"features_segments": new_segments, "is_anomaly_segment":is_anomaly_segment},
                features_path
            )
            del new_segments, is_anomaly_segment

            gc.collect() # Force Garbage Collection

            progress_bar.update(n=1)
    progress_bar.close()

    print()
    print("SUCCESS: Feature Extraction Completed.")

    cfg.DATA.READ_FEATURES = temp_read_features
    cfg.EXTRACT.ENABLE = temp_extract_enabled
    cfg.BACKBONE.TRAINABLE = temp_backbone_trainable


def _print_test_stats(cfg):
    """
    Prints a summary of the training process
    Args:
        cfg (cfgNode): Video model configurations
    """
    backbone_cfg = backbone_helper.get_backbone_merged_cfg(cfg)

    print("Training Summary:")

    headers = ["Attribute", "Value"]
    table = [
        ["Features Name", infoutils.get_dataset_features_name(cfg)],
    ]

    table = [x for x in table if x is not None]

    print(tabulate(table, headers, tablefmt="pretty", colalign=("center", "left")))
    print()
