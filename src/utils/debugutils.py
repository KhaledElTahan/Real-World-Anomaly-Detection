"""Debugging utilities"""

from pprint import pprint
import torch


def _tesnors_to_shapes(tensors):
    """
    Converts a nested list/tuple of torch tensors to list of shapes
    Args:
        tensors (list/tuple): Nested list of torch tensors
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
        elif isinstance(item, torch.Tensor):
            res = str(item.shape)
        else:
            res = item

        if isinstance(tensors, list):
            shapes = shapes + [res]
        elif isinstance(tensors, tuple):
            shapes = shapes + (res,)

    return shapes


def tensors_to_shapes(tensors):
    """
    Converts a nested list/tuple of torch tensors to list of shapes
    or a signle tensor to single shape
    Args:
        tensors: Nested list/tuple of torch tensors or a signle torch tensor
    Returns:
        shapes_list: a shape string or List of the same format as tensors
            but every tensor is replaced by its str(tensor.shape)
    """
    if isinstance(tensors, torch.Tensor):
        return str(tensors.shape)
    elif isinstance(tensors, list) or isinstance(tensors, tuple):
        return _tesnors_to_shapes(tensors)

    # undefinted, just ignore
    return tensors


def print_tensors_nicely(tensors):
    """
    Prints a nice format of the tensors or tensors list
    Args:
        tensors:  Nested list of torch tensors or a signle torch tensor
    """
    pprint(tensors_to_shapes(tensors))
