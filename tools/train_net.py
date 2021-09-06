"""Extract video features from the dataset using the backbone model."""

import gc
import time
import torch
from tabulate import tabulate
from tqdm import tqdm

from src.models import backbone_helper
from src.datasets import utils
from src.utils import infoutils
from src.utils import pathutils
from src.datasets import loader
from src.models.build import build_model
from src.models import losses
from src.engine import test_engine

def train(cfg):
    """
    Main tool for model training
    Args:
        cfg (cfgNode): Video Model Configurations
    """
    temp_read_features = cfg.DATA.READ_FEATURES
    temp_training_enabled = cfg.TRAIN.ENABLE

    cfg.TRAIN.ENABLE = True 
    cfg.DATA.READ_FEATURES = True # Force read features

    train_dataloader = loader.DatasetLoader(
        cfg, "train", cfg.DATA.READ_FEATURES,
        cfg.TRAIN.DATA_READ_ORDER, cfg.TRAIN.BATCH_SIZE, False
    )
    test_dataloader = loader.DatasetLoader(
        cfg, "test", cfg.DATA.READ_FEATURES,
        "Sequential", cfg.TEST.BATCH_SIZE, False
    )

    _print_train_stats(cfg)

    model = build_model(cfg)


    test_engine.test(model, test_dataloader, True)

    exit()

    for epoch in range(100):
        model.train()

        for normal_batch, anomaly_batch in train_dataloader:
            normal_output = model(normal_batch["features_batched"])
            anomaly_output = model(anomaly_batch["features_batched"])

            loss = losses.SultaniLoss(normal_output, anomaly_output)
            our_loss = loss()


        gc.collect()

    print()
    print("SUCCESS: Training Completed.")

    cfg.DATA.READ_FEATURES = temp_read_features
    cfg.TRAIN.ENABLE = temp_training_enabled


def _print_train_stats(cfg):
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
