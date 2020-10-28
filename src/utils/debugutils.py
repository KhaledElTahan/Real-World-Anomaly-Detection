"""Debugging utilities"""

from pprint import pprint
import torch


def tensors_list_to_shapes_list(tensors_list):
    """
    Converts a nested list/tuple of torch tensors to list of shapes
    Args:
        tensors_list (list/tuple): Nested list of torch tensors
    Returns:
        shapes_list (list): List of same format as tensors_list
            but every tensor is replaced by its str(tensor.shape)
    """
    shapes_list = []
    for item in tensors_list:
        if isinstance(item, list) or isinstance(item, tuple):
            shapes_list.append(tensors_list_to_shapes_list(item))
        elif isinstance(item, torch.Tensor):
            shapes_list.append(str(item.shape))
        else:
            shapes_list.append(item)

    return shapes_list


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
        return tensors_list_to_shapes_list(tensors)

    # undefinted, just ignore
    return tensors


def print_tensors_nicely(tensors):
    """
    Prints a nice format of the tensors or tensors list
    Args:
        tensors:  Nested list of torch tensors or a signle torch tensor
    """
    pprint(tensors_list_to_shapes_list(tensors))
