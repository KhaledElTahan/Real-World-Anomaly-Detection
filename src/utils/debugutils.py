"""Debugging utilities"""

from pprint import pprint
import torch


def tensors_list_to_sizes_list(tensors_list):
    """
    Converts a nested list of torch tensors to list of sizes
    Args:
        tensors_list (list): Nested list of torch tensors
    Returns:
        sizes_list (list): List of same format as tensors_list
            but every tensor is replaced by its str(tensor.shape)
    """
    sizes_list = []
    for item in tensors_list:
        if isinstance(item, list):
            sizes_list.append(tensors_list_to_sizes_list(item))
        elif isinstance(item, torch.Tensor):
            sizes_list.append(str(item.shape))
        else:
            sizes_list.append(item)

    return sizes_list


def print_tensors_nicely(tensors_list):
    """
    Prints a nice format of the tensors or tensors list
    Args:
        tensors_list (list): Nested list of torch tensors
    """
    if isinstance(tensors_list, torch.Tensor):
        print(tensors_list.shape)
    else:
        pprint(tensors_list_to_sizes_list(tensors_list))
