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
