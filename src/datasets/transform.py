"""Util Video Transformations"""
import torch
import cv2 as cv


def spatial_resize(images, new_height, new_width):
    """
    Resizes frames's spatial height and width
    Args:
        images (tensor): Images to perform resize. Dimension is
            `num frames` x `channel` x `height` x `width`.
        new_height (int): The new height
        new_width (int): The new width
    Returns:
        (tensor): The resized images with dimension of
            `num frames` x `channel` x `new height` x `new width`.
    """
    return torch.nn.functional.interpolate(
        images,
        size=(new_height, new_width),
        mode="bilinear",
        align_corners=False,
    )


def background_subtration(frames, algorithm):
    """
    Apply Backgroung subtraction on the frames
    Args:
        frames (list(np.ndarray)): List of np frames, each frame on HWC format
        algorithm (Str): BG Subtraction Algorithm, either 'MOG2' or 'KNN'
    Returns"
        frames (list(np.ndarray)): List of np frames, with background subtracted
    """
    assert algorithm in ['MOG2', 'KNN']

    if algorithm == 'MOG2':
        back_sub = cv.createBackgroundSubtractorMOG2()
    else:
        back_sub = cv.createBackgroundSubtractorKNN()

    for idx, frame in enumerate(frames):
        fg_mask = back_sub.apply(frame)
        res_frame = cv.bitwise_and(frame, frame, mask = fg_mask)
        frames[idx] = res_frame

    return frames


def tensor_color_scale_down(tensor):
    """
    Changes range or colors from [0, 255] to [0, 1]
    NOTE: Changes tensor type from torch.uint8 to torch.float32
    Args:
        tensor (Tensor(torch.uint8)): tensor to scale down
    Returns:
        tensor (Tensor(torch.float32)): scaled down tensor
    """
    assert tensor.dtype == torch.uint8

    tensor = tensor.float()
    tensor /= 255.0

    return tensor


def tensor_normalize(tensor, mean, std):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor or list): mean value to subtract.
        std (tensor or list): std to divide.
    """
    assert tensor.dtype == torch.float32

    if isinstance(mean, list):
        mean = torch.tensor(mean)
    if isinstance(std, list):
        std = torch.tensor(std)

    tensor -= mean
    tensor /= std
    return tensor


def revert_tensor_normalize(tensor, mean, std):
    """
    Revert normalization for a given tensor by multiplying by the std and adding the mean.
    Args:
        tensor (tensor): tensor to revert normalization.
        mean (tensor or list): mean value to add.
        std (tensor or list): std to multiply.
    """
    assert tensor.dtype == torch.float32

    if isinstance(mean, list):
        mean = torch.tensor(mean)
    if isinstance(std, list):
        std = torch.tensor(std)

    tensor = tensor * std
    tensor = tensor + mean
    return tensor
