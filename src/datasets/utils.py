"""Dataset Utils"""
import math
import numpy as np
import random
import time
import cv2
import torch
from fvcore.common.file_io import PathManager

from src.datasets import transform as transform
from src.models import backbone_helper as backbone_helper


def pack_pathway_output(cfg, frames):
    """
    Prepare output as a list of tensors. Each tensor corresponding to a
    unique pathway.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `channel` x `num frames` x `height` x `width`.
        cfg: The video model configuration file
    Returns:
        frame_list (list): list of tensors with the dimension of
            `channel` x `num frames` x `height` x `width`.
    """

    # First, retrieve the backbone configurations file
    backbone_cfg = backbone_helper.get_backbone_merged_cfg(cfg)

    if cfg.DATA.REVERSE_INPUT_CHANNEL:
        frames = frames[[2, 1, 0], :, :, :]
    if backbone_cfg.MODEL.ARCH in backbone_cfg.MODEL.SINGLE_PATHWAY_ARCH:
        frame_list = [frames]
    elif backbone_cfg.MODEL.ARCH in backbone_cfg.MODEL.MULTI_PATHWAY_ARCH:
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // backbone_cfg.SLOWFAST.ALPHA
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
    else:
        raise NotImplementedError(
            "Model arch {} is not in {}".format(
                backbone_cfg.MODEL.ARCH,
                backbone_cfg.MODEL.SINGLE_PATHWAY_ARCH + backbone_cfg.MODEL.MULTI_PATHWAY_ARCH,
            )
        )
    return frame_list


def _frames_to_frames_batches_native(frames :torch.Tensor, batch_size):
    """
    Converts tensor of `channel` x `num frames` x `height` x `width` to list of 
    len = `frames / batch size ` of tensors `channel` x `batch size` x `height` x `width`.
    Args:
        frames (torch.Tensor): the frames tensor of format (c, t, h, w)
        batch_size: The size of frames batches
    Return:
        frames_batches (list(list(torch.Tensor))): The batches of frames
        num_batches (Int): The number of frames batches, i.e. frames/batch_size
    """
    frames_batches = list(torch.split(frames, batch_size, dim=1))

    return [frames_batches], len(frames_batches)


def frames_to_frames_batches(cfg, frames):
    """
    Receives list of tensors of frames, either [frames] or [slow_pathway, fast_pathway]
    then converts each frames tensor from `channel` x `num frames` x `height` x `width` to
    list of len = `frames / batch size ` of tensors `channel` x `batch size` x `height` x `width`.
    Args:
        cfg (cfgNode): Video Model Configuration
        frames (list(torch.Tensor)): List of frames from dataset __getitem__
            on the form of [frames] or [slow_pathway, fast_pathway]
    Return:
        frames_batches list((list(torch.Tensor))): The batches of frames
            on the form of [frames_batches] or [slow_batches, fast_batches]
        num_batches (Int): The number of frames batches, i.e. frames/batch_size
    """
    # First, retrieve the backbone configurations file
    backbone_cfg = backbone_helper.get_backbone_merged_cfg(cfg)

    if backbone_cfg.MODEL.ARCH in backbone_cfg.MODEL.SINGLE_PATHWAY_ARCH:
        frames_batches, num_batches = _frames_to_frames_batches_native(frames,
            cfg.EXTRACT.FRAMES_BATCH_SIZE)
    elif backbone_cfg.MODEL.ARCH in backbone_cfg.MODEL.MULTI_PATHWAY_ARCH:
        slow_batches, num_slow_batches = _frames_to_frames_batches_native(
            frames[0], int(cfg.EXTRACT.FRAMES_BATCH_SIZE / backbone_cfg.SLOWFAST.ALPHA))
        fast_batches, num_fast_batches = _frames_to_frames_batches_native(frames[1],
            cfg.EXTRACT.FRAMES_BATCH_SIZE)

        # Assume number of fast frames is 257 -> ceil (257/16) = 17
        # Assume SlowFast.Alpha is 4
        # Then, number of slow frames is 64 -> ceil(64/(16/4)) = 16
        if num_fast_batches > num_slow_batches:
            fast_batches[0] = fast_batches[0][:-1]

        frames_batches, num_batches = [slow_batches[0], fast_batches[0]], num_slow_batches
    else:
        raise NotImplementedError(
            "Model arch {} is not in {}".format(
                backbone_cfg.MODEL.ARCH,
                backbone_cfg.MODEL.SINGLE_PATHWAY_ARCH + backbone_cfg.MODEL.MULTI_PATHWAY_ARCH,
            )
        )
    return frames_batches, num_batches


def frames_to_batches_of_frames_batches(cfg, frames):
    """
    Receives list of tensors of frames, either [frames] or [slow_pathway, fast_pathway]
    then converts each frames tensor from `channel` x `num frames` x `height` x `width` to
    list of len = `frames / batch size ` of tensors `channel` x `batch size` x `height` x `width`.
    Args:
        cfg (cfgNode): Video Model Configuration
        frames (list(torch.Tensor)): List of frames from dataset __getitem__
            on the form of [frames] or [slow_pathway, fast_pathway]
    Return:
        frames_batches list((list(torch.Tensor))): The batches of frames
            on the form of [frames_batches] or [slow_batches, fast_batches]
        num_batches (Int): The number of frames batches, i.e. frames/batch_size
    """
    # First, retrieve the backbone configurations file
    backbone_cfg = backbone_helper.get_backbone_merged_cfg(cfg)

    pass


def spatial_sampling(
    frames,
    spatial_idx=-1,
    min_scale=256,
    max_scale=320,
    crop_size=224,
    random_horizontal_flip=True,
    inverse_uniform_sampling=False,
):
    """
    Perform spatial sampling on the given video frames. If spatial_idx is
    -1, perform random scale, random crop, and random flip on the given
    frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
    with the given spatial_idx.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `num frames` x `height` x `width` x `channel`.
        spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
            or 2, perform left, center, right crop if width is larger than
            height, and perform top, center, buttom crop if height is larger
            than width.
        min_scale (int): the minimal size of scaling.
        max_scale (int): the maximal size of scaling.
        crop_size (int): the size of height and width used to crop the
            frames.
        inverse_uniform_sampling (bool): if True, sample uniformly in
            [1 / max_scale, 1 / min_scale] and take a reciprocal to get the
            scale. If False, take a uniform sample from [min_scale,
            max_scale].
    Returns:
        frames (tensor): spatially sampled frames.
    """
    assert spatial_idx in [-1, 0, 1, 2]
    if spatial_idx == -1:
        frames, _ = transform.random_short_side_scale_jitter(
            images=frames,
            min_size=min_scale,
            max_size=max_scale,
            inverse_uniform_sampling=inverse_uniform_sampling,
        )
        frames, _ = transform.random_crop(frames, crop_size)
        if random_horizontal_flip:
            frames, _ = transform.horizontal_flip(0.5, frames)
    else:
        # The testing is deterministic and no jitter should be performed.
        # min_scale, max_scale, and crop_size are expect to be the same.
        assert len({min_scale, max_scale, crop_size}) == 1
        frames, _ = transform.random_short_side_scale_jitter(
            frames, min_scale, max_scale
        )
        frames, _ = transform.uniform_crop(frames, crop_size, spatial_idx)
    return frames


def as_binary_vector(labels, num_classes):
    """
    Construct binary label vector given a list of label indices.
    Args:
        labels (list): The input label list.
        num_classes (int): Number of classes of the label vector.
    Returns:
        labels (numpy array): the resulting binary vector.
    """
    label_arr = np.zeros((num_classes,))

    for lbl in set(labels):
        label_arr[lbl] = 1.0
    return label_arr


def tensor_normalize(tensor, mean, std):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor or list): mean value to subtract.
        std (tensor or list): std to divide.
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    if type(mean) == list:
        mean = torch.tensor(mean)
    if type(std) == list:
        std = torch.tensor(std)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor


def revert_tensor_normalize(tensor, mean, std):
    """
    Revert normalization for a given tensor by multiplying by the std and adding the mean.
    Args:
        tensor (tensor): tensor to revert normalization.
        mean (tensor or list): mean value to add.
        std (tensor or list): std to multiply.
    """
    if type(mean) == list:
        mean = torch.tensor(mean)
    if type(std) == list:
        std = torch.tensor(std)
    tensor = tensor * std
    tensor = tensor + mean
    return tensor


def loader_worker_init_fn(dataset):
    """
    Create init function passed to pytorch data loader.
    Args:
        dataset (torch.utils.data.Dataset): the given dataset.
    """
    return None


def video_name_to_features_name(video_name, old_ext, new_ext):
    """
    Changes video file name to features file name with new extension
    Args:
        old_ext (String): The old extension of the video file
        new_ext (String): The new extension of the features file
    Examples:
        video_name_to_features_name("video.mp4 label 1 1", "mp4", "rar") -> "video.rar label 1 1"
    """
    return video_name.replace(old_ext, new_ext)