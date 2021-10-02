"""Utility used to simplify dev testing"""

def dev_test(cfg):
    """
    Used to simplify testing for developers, untracked on git
    NOTE: You must run the following git command in the project directory to be effective
        git update-index --assume-unchanged tools/dev_test.py
    NOTE: To undo the above command and make changes noticeable by git
        (to make changes for the config functionalty and not the actual configs)
        git update-index --no-assume-unchanged tools/dev_test.py
    """
    pass
