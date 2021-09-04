"""Debugging utilities"""

from pprint import pprint
import torch
import numpy as np


def _reduce_consec_strings(items):
    """
    Reduce consecutive strings inside a list/tuple, to beautify the output
    Args:
        items (list/tuple): Nested list of strings
    Returns:
        reduced_items (list/tuple): Same as items but with reduced elements
    Example:
        items: ('A', 'A', 'B', 'A')
        reduced_items: ('A x 2', 'B', 'A')
    """
    if isinstance(items, list):
        reduced_items = []
    elif isinstance(items, tuple):
        reduced_items = ()
    else:
        return items

    counter = 1
    for idx, item in enumerate(items):
        if isinstance(item, str) and idx + 1 < len(items) and item == items[idx + 1]:
            counter = counter + 1
        else:
            if counter > 1:
                res = "{} x {}".format(item, counter)
                counter = 1
            else:
                res = item

            if isinstance(items, list):
                reduced_items = reduced_items + [res]
            elif isinstance(items, tuple):
                reduced_items = reduced_items + (res,)

    return reduced_items


def _tesnors_to_shapes(tensors):
    """
    Converts a nested list/tuple/dict of torch/np tensors to list of shapes
    Args:
        tensors (list/tuple/dict): Nested list of torch/np tensors
    Returns:
        shapes (list/tuple/dict): List of same format as tensors_list
            but every tensor is replaced by its str(tensor.shape)
    """
    if isinstance(tensors, list):
        shapes = []
    elif isinstance(tensors, tuple):
        shapes = ()
    elif isinstance(tensors, dict):
        shapes = {}
        keys = list(tensors.keys())
        tensors = list(tensors.values())

    for idx, item in enumerate(tensors):
        if isinstance(item, (list, tuple, dict)):
            res = _tesnors_to_shapes(item)
        elif isinstance(item, (torch.Tensor, np.ndarray)):
            res = _tensor_representation(item)
        else:
            res = item

        if isinstance(shapes, list):
            shapes = shapes + [res]
        elif isinstance(shapes, tuple):
            shapes = shapes + (res,)
        elif isinstance(shapes, dict):
            shapes[keys[idx]] = res

    return _reduce_consec_strings(shapes)


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
    elif isinstance(tensors, (list, tuple, dict)):
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
