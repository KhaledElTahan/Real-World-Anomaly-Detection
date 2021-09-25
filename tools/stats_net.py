"""Print stats of the model"""

from tabulate import tabulate

from src.utils import pathutils
from src.utils import checkpoint


def stats(cfg):
    """
    Main tool for printing stats about the network
    Args:
        cfg (cfgNode): Video Model Configurations
    """
    checkpoints_dir_path = pathutils.get_models_checkpoint_directory_path(cfg)

    table = []
    for checkpoint_path in checkpoints_dir_path.iterdir():
        if checkpoint_path.is_file():
            _, _, auc, epoch, _, best_auc, best_epoch = checkpoint.load_checkpoint(
                cfg, False, checkpoint_path
            )
            table.append([checkpoint_path.stem, auc, best_auc, epoch, best_epoch])

    _print_trained_models_stats(cfg, table)


def _print_trained_models_stats(cfg, table):
    """
    Prints stats of the trained models
    Args:
        cfg (cfgNode): Video model configurations
        table (List): Table of data to be printed
    """

    print("Stats Summary:")

    headers = ["Model Name", "Last AUC", "Best AUC", "Epochs Completed", "Best Epoch"]

    print(tabulate(table, headers, tablefmt="pretty", colalign=("center", "center", "center", "center")))
    print()
