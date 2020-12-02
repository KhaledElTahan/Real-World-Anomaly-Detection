"""Debugging utilities"""

from pprint import pprint
import torch
import numpy as np


def _tesnors_to_shapes(tensors):
    """
    Converts a nested list/tuple of torch/np tensors to list of shapes
    Args:
        tensors (list/tuple): Nested list of torch/np tensors
    Returns:
        shapes (list/tuple): List of same format as tensors_list
            but every tensor is replaced by its str(tensor.shape)
    """
    if isinstance(tensors, list):
        shapes = []
    elif isinstance(tensors, tuple):
        shapes = ()

    for item in tensors:
        if isinstance(item, (list, tuple)):
            res = _tesnors_to_shapes(item)
        elif isinstance(item, (torch.Tensor, np.ndarray)):
            res = _tensor_representation(item)
        else:
            res = item

        if isinstance(tensors, list):
            shapes = shapes + [res]
        elif isinstance(tensors, tuple):
            shapes = shapes + (res,)

    return shapes


def _tensor_representation(tensor):
    """
    Creats a unified tensor representation
    Args:
        Tensor (Torch.Tensor or np.ndarray)
    Returns:
        representation (Str): Tensor representation
    """

    assert isinstance(tensor, (torch.Tensor, np.ndarray))

    representation = ''
    if isinstance(tensor, torch.Tensor):
        representation += 'Torch.Tensor('
    elif isinstance(tensor, np.ndarray):
        representation += 'Numpy.ndarray('

    representation += 'size=' + str(list(tensor.shape)) + ','
    representation += ' Type='
    representation += str(tensor.dtype) + ')'

    return representation


def tensors_to_shapes(tensors):
    """
    Converts a nested list/tuple of torch/numpy tensors to list of shapes
    or a signle tensor to single shape
    Args:
        tensors: Nested list/tuple of torch/np tensors or a signle torch/np tensor
    Returns:
        shapes_list: a shape string or List of the same format as tensors
            but every tensor is replaced by its str(tensor.shape)
    """
    if isinstance(tensors, (torch.Tensor, np.ndarray)):
        return _tensor_representation(tensors)
    elif isinstance(tensors, (list, tuple)):
        return _tesnors_to_shapes(tensors)

    # undefinted, just ignore
    return tensors


def print_tensors_nicely(tensors):
    """
    Prints a nice format of the tensors or tensors list
    Args:
        tensors:  Nested list of torch/np tensors or a signle torch/np tensor
    """
    pprint(tensors_to_shapes(tensors))
