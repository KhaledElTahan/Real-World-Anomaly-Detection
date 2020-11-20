"""Argument parser functions."""

import argparse
import sys

from src.config.defaults import get_cfg


def parse_args(): # needs an update
    """
    Parse the following arguments for a default parser for users.
    Args:
        gpus (int): number of used gpus, if 0 then use cpu
        cfg (str): path to the config file.
        init_method (str): initialization method to launch the job with multiple
            devices. Options includes TCP or shared file-system for
            initialization. details can be find in
            https://pytorch.org/docs/stable/distributed.html#tcp-initialization
        opts (argument): provide addtional options from the command line, it
            overwrites the config loaded from file.
        """
    parser = argparse.ArgumentParser(
        description="Provide model evaluation, feature extraction, training and testing pipeline."
    )
    parser.add_argument(
        "--gpus",
        help="Number of GPUs using by the job",
        default=0,
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
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        # default="configs/Kinetics/SLOWFAST_4x16_R50.yaml",
        type=str,
    )
    parser.add_argument(
        "--init_method",
        help="Initialization method, includes TCP or shared file-system",
        default="tcp://localhost:9999",
        type=str,
    )
    parser.add_argument(
        "--shard_id",
        help="The shard id of current node, Starts from 0 to num_shards - 1",
        default=0,
        type=int,
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

    # Load config from cfg.
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)

    # Load config from command line, overwrite config from opts.
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    # Inherit parameters from args.
    if hasattr(args, "gpus") :
        cfg.NUM_GPUS = args.gpus
    if hasattr(args, "rng_seed"):
        cfg.RNG_SEED = args.rng_seed
    if hasattr(args, "output_dir"):
        cfg.OUTPUT_DIR = args.output_dir
    if hasattr(args, "extract"):
        cfg.EXTRACT.ENABLE = args.extract

    return cfg
