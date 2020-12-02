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
