"""Set up Environment."""

_ENVIRONMENT_SETUP_DONE = False

def setup_environment():
    global _ENVIRONMENT_SETUP_DONE
    if _ENVIRONMENT_SETUP_DONE:
        return
    _ENVIRONMENT_SETUP_DONE = True
