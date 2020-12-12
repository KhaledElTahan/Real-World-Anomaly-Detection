"""Wrapper to use the video model."""
import sys
from pathlib import Path
app_directory = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(app_directory))
from src.utils.parser import load_config, parse_args
from tools.extract_video_features import extract

def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    cfg = load_config(args)

    # --extract
    # Perform feature extraction
    if cfg.EXTRACT.ENABLE:
        extract(cfg)

    print()
    print("Execution completed.")


if __name__ == "__main__":
    main()
