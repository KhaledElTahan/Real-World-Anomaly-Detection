"""Argument parser functions."""

import argparse
import sys

from src.config.defaults import get_cfg
from src.utils import pathutils
from src.config import custom_untracked_config, custom_tracked_config


def parse_args():
    """
    Parse the following arguments for a default parser for users.
    Args:
        gpus (int): Number of used gpus, if 0 then use cpu.
        extract (Bool): Whether to extract features or not.
        trainbs (int): Extraction batch size
        train (Bool): Whether to train the model or not.
        trainbs (int): Train batch size.
        test (Boool): Whether to evaluate the model or not.
        testbs (int): Test batch size.
        readorder (str): Training data read order.
        cfg_extract (str): path to the feature extraction config file.
        opts (argument): provide addtional options from the command line, it
            overwrites the config loaded from file.
        """
    parser = argparse.ArgumentParser(
        description="Provide model evaluation, feature extraction, training and testing pipeline."
    )
    parser.add_argument(
        "--gpus",
        help="Number of GPUs using by the job",
        default=-1,
        type=int,
    )
    parser.add_argument(
        "--extract",
        dest="extract",
        help="Select this option to extract the features",
        action='store_true',
    )
    parser.add_argument(
        "--extractbs",
        dest="extractbs",
        help="Extraction batch size",
        default=-1,
        type=int,
    )
    parser.add_argument(
        "--train",
        dest="train",
        help="Select this option to train the model",
        action='store_true',
    )
    parser.add_argument(
        "--trainbs",
        dest="trainbs",
        help="Training batch size",
        default=-1,
        type=int,
    )
    parser.add_argument(
        "--test",
        dest="test",
        help="Select this option to test and evaluate the model",
        action='store_true',
    )
    parser.add_argument(
        "--testbs",
        dest="testbs",
        help="Testing batch size",
        default=-1,
        type=int,
    )
    parser.add_argument(
        "--readorder",
        dest="readorder",
        help="Specify the training read order",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--stats",
        dest="stats",
        help="Select this option to run the stats tool",
        action='store_true',
    )
    parser.add_argument(
        "--devtest",
        dest="devtest",
        help="Select this option to run the developer testing tool",
        action='store_true',
    )
    parser.set_defaults(extract=False)
    parser.add_argument(
        "--cfg_extract",
        dest="extraction_cfg_file",
        help="Path to the config file",
        default="features-configs/UCFAnomalyDetection/Kinetics_c2_SLOWFAST_8x8_R50_NONE_32x32.yaml",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="See src/config/defaults.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )

    if '-h' in sys.argv or '--help' in sys.argv:
        parser.print_help()

        if len(sys.argv) == 2:
            sys.exit()

    return parser.parse_args()

def load_config(args):
    """
    Given the arguemnts, load and initialize the configs.
    Args:
        args (argument): arguments includes `gpus`, `cfg_file`,
            `init_method` , and `opts`.
    """
    # Setup cfg.
    cfg = get_cfg()

    # Load extraction config into cfg.
    if args.extraction_cfg_file is not None:
        extraction_cfg_file_path = pathutils.get_configs_path() / args.extraction_cfg_file
        cfg.merge_from_file(str(extraction_cfg_file_path))

    # Check whether we have to extract features
    cfg.EXTRACT.ENABLE = args.extract

    # Check whether we have to train model
    cfg.TRAIN.ENABLE = args.train

    # Check whether we have to test model
    cfg.TEST.ENABLE = args.test

    # Check whether we have to run the stats tool
    cfg.RUN_STATS_TOOL = args.stats

    # Check whether we have to run the stats tool
    cfg.RUN_DEV_TEST = args.devtest

    # Set number of GPUs
    if args.gpus != -1:
        cfg.NUM_GPUS = args.gpus

    if args.readorder is not None:
        cfg.TRAIN.DATA_READ_ORDER = args.readorder

    if args.extractbs != -1:
        cfg.EXTRACT.FRAMES_BATCHES_BATCH_SIZE = args.extractbs

    if args.trainbs != -1:
        cfg.TRAIN.BATCH_SIZE = args.trainbs

    if args.testbs != -1:
        cfg.TEST.BATCH_SIZE = args.testbs

    # Load config from command line, overwrite config from opts.
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    # Final Step, add custom config with default values.
    custom_tracked_config.add_custom_tracked_config(cfg)
    custom_untracked_config.add_custom_untracked_config(cfg)

    return cfg
