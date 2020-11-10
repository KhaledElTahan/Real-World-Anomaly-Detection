"""Extract video features from the dataset using the backbone model."""

import gc
import torch
from tabulate import tabulate
from tqdm import tqdm
import time

from src.models import backbone_helper
from src.datasets import utils
from src.datasets import build
from src.utils import modelutils


@torch.no_grad()
def extract(cfg):
    """
    Main tool for feature extraction
    Args:
        cfg (cfgNode): Video Model Configurations
    """
    temp_read_features = cfg.DATA.READ_FEATURES
    temo_extract_enabled = cfg.EXTRACT.ENABLE

    cfg.EXTRACT.ENABLE = True # Force train dataset to get items without respect to anomalies
    cfg.DATA.READ_FEATURES = False # Force read videos

    datasets = []
    for split in cfg.EXTRACT.DATASET_SPLITS:
        datasets.append(build.build_dataset(cfg.EXTRACT.DATASET, cfg, split))

    backbone_model = backbone_helper.load_model(cfg)
    backbone_model.eval()

    features_length = modelutils.get_features_length(cfg, backbone_model)

    if features_length != cfg.BACKBONE.FEATURES_LENGTH:
        print("Warning: Set cfg.BACKBONE.FEATURES_LENGTH with value {}", features_length)

    total_len = sum([len(dataset) for dataset in datasets])

    _print_extract_stats(cfg, features_length, total_len)

    progress_bar = tqdm(total=total_len, desc="Feature Extraction Progress")
    for dataset in datasets:
        for _, (frames, _, annotation, video_index) in enumerate(dataset):

            features_path = utils.video_path_to_features_path(
                cfg, dataset.get_video_path(video_index)
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
    cfg.EXTRACT.ENABLE = temo_extract_enabled


def _print_extract_stats(cfg, features_length, videos_num):
    """
    Prints a summary of the extraction process
    Args:
        cfg (cfgNode): Video model configurations
        features_length (Int): Length of extracted features dimension
        videos_num (Int): Number of videos that will be processed
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
        ["Number of datasets' videos", videos_num],
    ]

    table = [x for x in table if x is not None]

    print(tabulate(table, headers, tablefmt="pretty", colalign=("center", "left")))
    print()
