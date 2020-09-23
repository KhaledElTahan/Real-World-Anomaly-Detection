"""Set up Environment."""
import sys
import src.utils.pathutils as pathutils

_ENVIRONMENT_SETUP_DONE = False

def setup_environment():
    global _ENVIRONMENT_SETUP_DONE
    if _ENVIRONMENT_SETUP_DONE:
        return

    sys.path.insert(0, str(pathutils.get_models_path()))

    _ENVIRONMENT_SETUP_DONE = True
