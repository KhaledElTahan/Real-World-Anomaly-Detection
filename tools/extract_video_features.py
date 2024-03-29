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


@torch.no_grad()
def extract(cfg):
    """
    Main tool for feature extraction
    Args:
        cfg (cfgNode): Video Model Configurations
    """
    temp_read_features = cfg.DATA.READ_FEATURES
    temp_extract_enabled = cfg.EXTRACT.ENABLE
    temp_backbone_trainable = cfg.BACKBONE.TRAINABLE


    cfg.EXTRACT.ENABLE = True
    cfg.DATA.READ_FEATURES = False # Force read videos
    cfg.BACKBONE.TRAINABLE = False # Force backbone to detach features

    datasets = []
    for split in cfg.EXTRACT.DATASET_SPLITS:
        datasets.append(build.build_dataset(cfg.EXTRACT.DATASET, cfg, split, cfg.DATA.READ_FEATURES))

    backbone_model = backbone_helper.load_model(cfg)
    backbone_model.eval()

    features_length = modelutils.get_features_length(cfg, backbone_model)

    if features_length != cfg.BACKBONE.FEATURES_LENGTH:
        print("Warning: Set cfg.BACKBONE.FEATURES_LENGTH with value ", features_length)

    total_len = sum([len(dataset) for dataset in datasets])

    _print_extract_stats(cfg, features_length, total_len)

    progress_bar = tqdm(total=total_len, desc="Feature Extraction Progress")
    for dataset in datasets:
        for _, (frames, _, _, annotation, video_path) in enumerate(dataset):

            features_path = pathutils.video_path_to_features_path(cfg, video_path)
            if not cfg.EXTRACT.FORCE_REWRITE and features_path.exists():
                progress_bar.update(n=1)
                time.sleep(0.05)
                continue

            frames_batches = utils.frames_to_batches_of_frames_batches(cfg, frames)
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
                cfg, torch.cat(features_batches), annotation
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


def _print_extract_stats(cfg, features_length, videos_num):
    """
    Prints a summary of the extraction process
    Args:
        cfg (cfgNode): Video model configurations
        features_length (Int): Length of extracted features dimension
        videos_num (Int): Number of videos that will be processed
    """
    backbone_cfg = backbone_helper.get_backbone_merged_cfg(cfg)

    print("Extraction Summary:")

    headers = ["Attribute", "Value"]
    table = [
        ["Features Name", infoutils.get_dataset_features_name(cfg)],
        ["Frames inner batch size", cfg.EXTRACT.FRAMES_BATCH_SIZE],
        ["Frames stack batch size", cfg.EXTRACT.FRAMES_BATCHES_BATCH_SIZE],
        ["Number of Output Segments", cfg.EXTRACT.NUMBER_OUTPUT_SEGMENTS],
        ["Video Scales", cfg.DATA.SCALES],
        ["Backbone", cfg.BACKBONE.NAME],
        ["SlowFast.Alpha", backbone_cfg.SLOWFAST.ALPHA]
            if backbone_cfg.MODEL.ARCH in backbone_cfg.MODEL.MULTI_PATHWAY_ARCH else None,
        ["Backbone Trainable", cfg.BACKBONE.TRAINABLE],
        ["Machine Type", "CPU" if cfg.NUM_GPUS == 0 else "GPU"],
        ["No. GPUs", cfg.NUM_GPUS],
        ["CFG. Features Length", cfg.BACKBONE.FEATURES_LENGTH],
        ["Actual Features Length", features_length],
        ["Number of datasets' videos", videos_num],
        ["Background Subtraction", str(cfg.TRANSFORM.BG_SUBTRACTION_ENABLED)],
        ["BG_Sub Algorithm", cfg.TRANSFORM.BG_SUBTRACTION_ALGORITHM]
            if cfg.TRANSFORM.BG_SUBTRACTION_ENABLED else None,
        ["Transformation Code", cfg.TRANSFORM.CODE],
    ]

    table = [x for x in table if x is not None]

    print(tabulate(table, headers, tablefmt="pretty", colalign=("center", "left")))
    print()
