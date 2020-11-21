"""Argument parser functions."""

import argparse
import sys

from src.config.defaults import get_cfg
from src.utils import pathutils
from src.config import custom_config


def parse_args(): 
    """
    Parse the following arguments for a default parser for users.
    Args:
        gpus (int): Number of used gpus, if 0 then use cpu.
        extract: Whether to extract features or not.
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

    # Check whether we have to extract features
    if hasattr(args, "extract"):
        cfg.EXTRACT.ENABLE = args.extract

    # Load extraction config into cfg.
    if cfg.EXTRACT.ENABLE and args.extraction_cfg_file is not None:
        extraction_cfg_file_path = pathutils.get_configs_path() / args.extraction_cfg_file
        cfg.merge_from_file(str(extraction_cfg_file_path))

    # Set number of GPUs
    if hasattr(args, "gpus") and args.gpus != -1:
        cfg.NUM_GPUS = args.gpus

    # Load config from command line, overwrite config from opts.
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    # Final Step, add custom config with default values.
    custom_config.add_custom_config(cfg)

    return cfg
