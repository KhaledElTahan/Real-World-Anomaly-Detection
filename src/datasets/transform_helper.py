"""Apply set of transformations using the configurations object"""
import src.datasets.transform as transform
from src.datasets import utils
from src.utils import funcutils


@funcutils.debug(apply=False, sign=True, ret=True, sign_beautify=True, ret_beautify=True)
def apply_transformations_list_np_frames(cfg, frames):
    """
    Apply a list of transformations on the list of np HWC frames according
    to the configurations file
    Args:
        cfg (CfgNode): Video model configurations file
        frames (list(np.ndarray)): List of np frames, each frame on HWC format
    """

    # Perform Background Subtraction
    if cfg.TRANSFORM.BG_SUBTRACTION_ENABLED:
        frames = transform.background_subtration(
            frames, cfg.TRANSFORM.BG_SUBTRACTION_ALGORITHM
        )

    return frames


@funcutils.debug(apply=False, sign=True, ret=True, sign_beautify=True, ret_beautify=True)
def apply_transformations_np_frames(cfg, frames):
    """
    Apply a list of transformations on the THWC np frames according to the configurations file
    Args:
        cfg (CfgNode): Video model configurations file
        frames (np.ndarray): np frames with THWC format
    """
    return frames


@funcutils.debug(apply=False, sign=False, ret=True, sign_beautify=True, ret_beautify=True)
def apply_transformations_THWC_torch_frames(cfg, frames):
    """
    Apply a list of transformations on the THWC Torch frames according to the configurations file
    Args:
        cfg (CfgNode): Video model configurations file
        frames (Torch.Tensor): Torch frames with THWC format
    """
    # Perform color normalization.
    frames = utils.tensor_normalize(frames, cfg.DATA.MEAN, cfg.DATA.STD)

    return frames


@funcutils.debug(apply=False, sign=False, ret=True, sign_beautify=True, ret_beautify=True)
def apply_transformations_CTHW_torch_frames(cfg, frames):
    """
    Apply a list of transformations on the CTHW Torch frames according to the configurations file
    Args:
        cfg (CfgNode): Video model configurations file
        frames (Torch.Tensor): Torch frames with CTHW format
    """
    # Spatial Scaling
    if frames.shape[2] != cfg.DATA.SCALES[0] or frames.shape[3] != cfg.DATA.SCALES[1]:
        frames = transform.spatial_resize(frames, cfg.DATA.SCALES[0], cfg.DATA.SCALES[1])

    return frames
