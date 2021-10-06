"""Wrapper to use the video model."""
import sys
from pathlib import Path
app_directory = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(app_directory))
from src.utils.parser import load_config, parse_args
from src.utils import env
from tools.extract_video_features import extract
from tools.train_net import train
from tools.test_net import test
from tools.demo_net import demo
from tools.stats_net import stats
from tools.dev_test import dev_test


def main():
    """
    Main function runner tool
    """
    args = parse_args()
    cfg = load_config(args)
    env.setup_random_environment(cfg)

    # --extract
    # Perform feature extraction
    if cfg.EXTRACT.ENABLE:
        extract(cfg)

    # --train
    # Train the model
    if cfg.TRAIN.ENABLE:
        train(cfg)

    # --test
    # Test the model
    if cfg.TEST.ENABLE:
        test(cfg)

    # --demo
    # Make a demo
    if cfg.DEMO.ENABLE:
        demo(cfg)

    # --stats
    # Print stats
    if cfg.RUN_STATS_TOOL:
        stats(cfg)

    # --devtest
    # Run development tests
    if cfg.RUN_DEV_TEST:
        dev_test(cfg)

    print()
    print("Execution completed.")


if __name__ == "__main__":
    main()
