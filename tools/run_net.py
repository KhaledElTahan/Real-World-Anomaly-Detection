"""Wrapper to use the video model."""
import sys
from pathlib import Path
app_directory = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(app_directory)) 
from src.utils.misc import launch_job
from src.utils.parser import load_config, parse_args
from tools.extract_video_features import extract

def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    cfg = load_config(args)

    # Perform feature extraction
    if cfg.EXTRACT.ENABLE:
         launch_job(cfg=cfg, init_method=args.init_method, func=extract)


if __name__ == "__main__":
    main()
