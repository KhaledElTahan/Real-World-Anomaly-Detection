"""Set up Environment."""
import sys
from pathlib import Path

_ENVIRONMENT_SETUP_DONE = False

def setup_environment():
    global _ENVIRONMENT_SETUP_DONE
    if _ENVIRONMENT_SETUP_DONE:
        return

    models_dir = Path(__file__).absolute().parents[1] / "models"
    sys.path.insert(0, str(models_dir)) 
    
    _ENVIRONMENT_SETUP_DONE = True
