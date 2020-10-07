"""Apply set of transformations using the configurations object"""
import src.datasets.transform as transform
import src.datasets.cv2_transform as cv2_transform


def apply_transformations(frames, cfg):
    """
    Apply a list of transformations on the frames according to the configurations file
    Args:
        cfg (CfgNode): Video model configurations file
    """
    return frames
    