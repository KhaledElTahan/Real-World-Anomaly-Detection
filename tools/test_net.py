"""Test the whole model."""

from tabulate import tabulate

from src.utils import infoutils, checkpoint
from src.datasets import loader
from src.models.build import build_model
from src.engine import test_engine
from src.visualization import roc_auc


def test(cfg):
    """
    Main tool for model testing
    Args:
        cfg (cfgNode): Video Model Configurations
    """
    temp_read_features = cfg.DATA.READ_FEATURES
    temp_testing_enabled = cfg.TEST.ENABLE

    cfg.TEST.ENABLE = True
    cfg.DATA.READ_FEATURES = True # Force read features

    test_dataloader = loader.DatasetLoader(
        cfg, "test", cfg.DATA.READ_FEATURES,
        "Sequential", cfg.TEST.BATCH_SIZE, False
    )

    assert checkpoint.checkpoint_exists(cfg), "Checkpoint is needed for Testing!"

    _, _, _, completed_epochs, best_model_state_dict, best_auc, _ = checkpoint.load_checkpoint(cfg)
    model = build_model(cfg, best_model_state_dict)

    _print_test_stats(cfg, completed_epochs, best_auc)

    auc, fpr, tpr, _ = test_engine.test(model, test_dataloader, True)
    roc_auc.plot_signle_roc_auc(cfg, auc, fpr, tpr)

    print()
    print("SUCCESS: Testing Completed.")

    cfg.DATA.READ_FEATURES = temp_read_features
    cfg.TRAIN.ENABLE = temp_testing_enabled


def _print_test_stats(cfg, completed_epochs, best_auc):
    """
    Prints a summary of the testing process
    Args:
        cfg (cfgNode): Video model configurations
        training_from_checkpoint (Bool): Whether the training is to continue
            from a checkpoint or not
        completed_epochs (int): Number of completed epochs if training_from_checkpoint
        best_auc (float): Best achieved AUC if training_from_checkpoint
    """

    print("Testing Summary:")

    headers = ["Attribute", "Value"]
    table = [
        ["Full Model Name", infoutils.get_full_model_name(cfg)],
        ["Classifier Name", cfg.MODEL.MODEL_NAME],
        ["Loss Name", cfg.MODEL.LOSS_FUNC],
        ["Dataset", cfg.TRAIN.DATASET],
        ["Test Batch Size", cfg.TEST.BATCH_SIZE],
        ["Number of Segments", cfg.EXTRACT.NUMBER_OUTPUT_SEGMENTS],
        ["Training Type", cfg.TRAIN.TYPE],
        ["Completed Epochs", completed_epochs],
        ["Best AUC", best_auc],
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
