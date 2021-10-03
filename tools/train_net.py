"""Train the whole model."""

import gc
from tabulate import tabulate

from src.utils import infoutils, checkpoint, modelutils
from src.datasets import loader
from src.models.build import build_model
from src.models import optimizers
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

    training_from_checkpoint = cfg.TRAIN.AUTO_RESUME and checkpoint.checkpoint_exists(cfg)
    if training_from_checkpoint:
        _, optimizer_state_dict, model_state_dict, auc, completed_epochs, \
        best_model_state_dict, best_auc, best_epoch = checkpoint.load_checkpoint(cfg)
        model = build_model(cfg, model_state_dict)
        optimizer = optimizers.get_optimizer(cfg, model)
        optimizer.load_state_dict(optimizer_state_dict)
    else:
        model = build_model(cfg)
        optimizer = optimizers.get_optimizer(cfg, model)
        completed_epochs = 0
        best_auc = 0.0
        best_epoch = 1

    _print_train_stats(
        cfg,
        training_from_checkpoint,
        completed_epochs,
        best_auc,
        train_dataloader.examples_length,
        len(train_dataloader)
    )

    for epoch in range(completed_epochs + 1, cfg.TRAIN.MAX_EPOCH + 1):
        cfg.TRAIN.CURRENT_EPOCH = epoch
        loss_value = train_engine.train(
            cfg,
            model,
            optimizer,
            train_dataloader,
            test_dataloader,
            epoch,
            True
        )

        if epoch % cfg.TRAIN.EVAL_PERIOD == 0:
            auc, _, _, _ = test_engine.test(cfg, model, test_dataloader, True)

            if auc > best_auc:
                best_auc = auc
                best_epoch = epoch
                best_model_state_dict = modelutils.create_state_dictionary(cfg, model)

            print("Loss = {0:.6f} --- AUC = {1:.6f} --- Best AUC = {2:.6f} -- Best Epoch = {3}\n".
                format(loss_value, auc, best_auc, best_epoch))

        if epoch % cfg.TRAIN.CHECKPOINT_PERIOD == 0 and best_auc != 0.0:
            checkpoint.save_checkpoint(
                cfg, optimizer, model, auc, best_model_state_dict, best_auc, best_epoch
            )

        gc.collect()

    print()
    print("SUCCESS: Training Completed.")

    cfg.DATA.READ_FEATURES = temp_read_features
    cfg.TRAIN.ENABLE = temp_training_enabled


def _print_train_stats(
        cfg, training_from_checkpoint, completed_epochs, best_auc, train_examples_len, train_batches_len
    ):
    """
    Prints a summary of the training process
    Args:
        cfg (cfgNode): Video model configurations
        training_from_checkpoint (Bool): Whether the training is to continue
            from a checkpoint or not
        completed_epochs (int): Number of completed epochs if training_from_checkpoint
        best_auc (float): Best achieved AUC if training_from_checkpoint
        train_examples_len (int): Number of training examples
        train_batches_len (int): Number of training batches
    """

    print("Training Summary:")

    headers = ["Attribute", "Value"]
    table = [
        ["Full Model Name", infoutils.get_full_model_name(cfg)],
        ["Classifier Name", cfg.MODEL.MODEL_NAME],
        ["Loss Name", cfg.MODEL.LOSS_FUNC],
        ["Dataset", cfg.TRAIN.DATASET],
        ["Train Batch Size", cfg.TRAIN.BATCH_SIZE],
        ["Test Batch Size", cfg.TEST.BATCH_SIZE],
        ["Number of Training Examples", train_examples_len],
        ["Number of Training Batches", train_batches_len],
        ["Number of Segments", cfg.EXTRACT.NUMBER_OUTPUT_SEGMENTS],
        ["Extraction Frames Inner Batch Size", cfg.EXTRACT.FRAMES_BATCH_SIZE],
        ["Training Type", infoutils.get_detailed_train_type(cfg)],
        ["Aug Dataset Transform Code", cfg.TRAIN.PL_AUG_CODE]
            if cfg.TRAIN.TYPE in ["PL", "PL-MIL"] else None,
        ["PL MIL Intervals", cfg.TRAIN.PL_MIL_INTERVALS]
            if cfg.TRAIN.TYPE == "PL-MIL" else None,
        ["PL MIL Percentages", cfg.TRAIN.PL_MIL_PERCENTAGES]
            if cfg.TRAIN.TYPE == "PL-MIL" else None,
        ["PL MIL - MIL First", cfg.TRAIN.PL_MIL_MILFIRST]
            if cfg.TRAIN.TYPE == "PL-MIL" else None,
        ["Training Data Read Order", cfg.TRAIN.DATA_READ_ORDER],
        ["Training from Checkpoint", training_from_checkpoint],
        ["Completed Epochs", completed_epochs] if training_from_checkpoint else None,
        ["Best AUC", best_auc] if training_from_checkpoint else None,
        ["Evaluate per epoch period", cfg.TRAIN.EVAL_PERIOD],
        ["Checkpoint per epoch period", cfg.TRAIN.CHECKPOINT_PERIOD],
        ["Evaluate inside training epoch", cfg.TRAIN.ENABLE_EVAL_BATCH],
        ["Evaluate per batch in epoch period", cfg.TRAIN.EVAL_BATCH_PERIOD]
            if cfg.TRAIN.ENABLE_EVAL_BATCH else None,
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
