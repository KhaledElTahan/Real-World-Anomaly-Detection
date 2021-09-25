"""Print stats of the model"""

from tabulate import tabulate

from src.utils import pathutils, infoutils, checkpoint


def stats(cfg):
    """
    Main tool for printing stats about the overall network
    Args:
        cfg (cfgNode): Video Model Configurations
    """
    _print_trained_models_stats(cfg)


def _print_trained_models_stats(cfg):
    """
    Prints stats of the trained models
    Args:
        cfg (cfgNode): Video model configurations
    """

    print("Trained Model Stats:")

    headers = [
        "Features Name",
        "Model Name",
        "Reading Order",
        "Best AUC",
        "Best Epoch",
        "Last AUC",
        "Epochs Completed",
    ]

    table = []
    for checkpoint_path in pathutils.get_all_checkpoints_paths(cfg):
        if checkpoint_path.is_file():
            checkpoint_cfg, _, _, auc, epoch, _, best_auc, best_epoch = checkpoint.load_checkpoint(
                cfg, False, checkpoint_path
            )

            table.append(
                [
                    infoutils.get_dataset_features_name(checkpoint_cfg),
                    infoutils.get_full_model_without_features(checkpoint_cfg),
                    checkpoint_cfg.TRAIN.DATA_READ_ORDER,
                    "{:6f}".format(best_auc),
                    best_epoch,
                    "{:6f}".format(auc),
                    epoch
                ]
            )

    table.sort(key=lambda row: row[1])
    table.sort(key=lambda row: row[0])

    spaced_table = []
    for index, row in enumerate(table):
        if index > 0 and table[index][0] != table[index - 1][0]:
            spaced_table.append(['---'] * len(headers))
        spaced_table.append(row)

    print(tabulate(
        spaced_table, headers, tablefmt="pretty",
        colalign=("center", "center", "center", "center", "center", "center", "center"))
    )

    print()
