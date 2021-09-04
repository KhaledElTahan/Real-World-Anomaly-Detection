"""Utilities related to the dataset loader"""

import torch
from src.utils import funcutils


@funcutils.debug(apply=False, sign=True, ret=True, sign_beautify=True, ret_beautify=True)
def features_dataset_results_list_to_batch(features_results):
    """
    Merges features dataset list of results into one batch
    Args:
        features_results (List): a list of features dataset output
    Returns
        batch (dict):
            {
                features_batched: Tensor(Torch) features batched,
                labels: List(str) labels batched,
                one_hots: Tensor(Torch) one hot vectors batched,
                annotations: Tensor(Torch) segments annotations batched,
                paths: List(Path) features paths batched
            }
    """
    features_list = []
    label_list = []
    one_hot_list = []
    annotation_list = []
    path_list = []

    for features, label, one_hot, annotation, path in features_results:
        features_list.append(features)
        label_list.append(label)
        one_hot_list.append(one_hot)
        annotation_list.append(annotation)
        path_list.append(path)

    batch = {}

    batch["features_batched"] = torch.stack(features_list, dim=0)
    batch["labels"] = label_list
    batch["one_hots"] = torch.stack(one_hot_list, dim=0)
    batch["annotations"] = torch.stack(annotation_list, dim=0)
    batch["paths"] = path_list

    return batch
