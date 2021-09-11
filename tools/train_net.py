"""Extract video features from the dataset using the backbone model."""

import gc
from tabulate import tabulate

from src.utils import infoutils
from src.datasets import loader
from src.models.build import build_model
from src.models import losses, optimizers
from src.engine import test_engine, train_engine

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
    optimizer = optimizers.get_optimizer(cfg, model)

    best_auc = 0.0
    for epoch in range(100):
        train_engine.train(model, losses.get_loss_class(cfg), optimizer, train_dataloader, epoch + 1, True)
        auc, _, _, _ = test_engine.test(model, test_dataloader, True)
        best_auc = max(best_auc, auc)
        print("Best AUC so far ", best_auc)
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

    print("Training Summary:")

    headers = ["Attribute", "Value"]
    table = [
        ["Full Model Name", infoutils.get_full_model_name(cfg)],
        ["Classifier Name", cfg.MODEL.MODEL_NAME],
        ["Loss Name", cfg.MODEL.LOSS_FUNC],
        ["Dataset", cfg.TRAIN.DATASET],
        ["Train Dataset Reading Order", cfg.TRAIN.DATA_READ_ORDER],
        ["Train Batch Size", cfg.TRAIN.BATCH_SIZE],
        ["Test Batch Size", cfg.TEST.BATCH_SIZE],
        ["Number of Segments", cfg.EXTRACT.NUMBER_OUTPUT_SEGMENTS],
        ["Training Type", cfg.TRAIN.TYPE],
        ["Features Name", infoutils.get_dataset_features_name(cfg)],
        ["Backbone", cfg.BACKBONE.NAME],
        ["Backbone Trainable", cfg.BACKBONE.TRAINABLE],
        ["Optimizer", cfg.OPTIMIZER.NAME],
        ["Base Learning Rate", cfg.OPTIMIZER.BASE_LR],
        ["Machine Type", "CPU" if cfg.NUM_GPUS == 0 else "GPU"],
        ["No. GPUs", cfg.NUM_GPUS],
        ["CFG. Features Length", cfg.BACKBONE.FEATURES_LENGTH],
        ["Background Subtraction", str(cfg.TRANSFORM.BG_SUBTRACTION_ENABLED)],
        ["BG_Sub Algorithm", cfg.TRANSFORM.BG_SUBTRACTION_ALGORITHM]
            if cfg.TRANSFORM.BG_SUBTRACTION_ENABLED else None,
        ["Transformation Code", cfg.TRANSFORM.CODE],
    ]

    table = [x for x in table if x is not None]

    print(tabulate(table, headers, tablefmt="pretty", colalign=("center", "left")))
    print()
