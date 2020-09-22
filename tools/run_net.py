"""Wrapper to use the video model."""
import sys
import os
sys.path.insert(0, os.path.abspath('..'))

from src.utils.misc import launch_job
from src.utils.parser import load_config, parse_args

def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    cfg = load_config(args)

    # Perform feature extraction
    # if cfg.EXTRACT.ENABLE:
    #     launch_job(cfg=cfg, init_method=args.init_method, func=extract)

    # Perform evaluation.
    # if cfg.EVAL.ENABLE:
    #    launch_job(cfg=cfg, init_method=args.init_method, func=train)

    # Perform training.
    # if cfg.TRAIN.ENABLE:
    #     launch_job(cfg=cfg, init_method=args.init_method, func=train)

    # Perform multi-clip testing.
    # if cfg.TEST.ENABLE:
    #    launch_job(cfg=cfg, init_method=args.init_method, func=test)




if __name__ == "__main__":
    main()
