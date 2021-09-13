"""Run a demo of the model"""

from tabulate import tabulate

from src.utils import infoutils


def demo(cfg):
    pass


def _print_demo_stats(cfg, completed_epochs, best_auc):
    """
    Prints a summary of the demo
    Args:
        cfg (cfgNode): Video model configurations
        completed_epochs (int): Number of completed epochs if training_from_checkpoint
        best_auc (float): Best achieved AUC if training_from_checkpoint
    """

    print("Demo Summary:")

    headers = ["Attribute", "Value"]
    table = [
        ["Full Model Name", infoutils.get_full_model_name(cfg)],
        ["Classifier Name", cfg.MODEL.MODEL_NAME],
        ["Loss Name", cfg.MODEL.LOSS_FUNC],
        ["Number of Segments", cfg.EXTRACT.NUMBER_OUTPUT_SEGMENTS],
        ["Extraction Frames Inner Batch Size", cfg.EXTRACT.FRAMES_BATCH_SIZE],
        ["Training Type", cfg.TRAIN.TYPE],
        ["Training Data Read Order", cfg.TRAIN.DATA_READ_ORDER],
        ["Completed Epochs", completed_epochs],
        ["Best AUC", best_auc],
        ["Features Name", infoutils.get_dataset_features_name(cfg)],
        ["Backbone", cfg.BACKBONE.NAME],
        ["Backbone Trainable", cfg.BACKBONE.TRAINABLE],
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
