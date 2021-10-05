"""Creates Datasets Registery & Dataset Builder"""
from fvcore.common.registry import Registry

DATASET_REGISTRY = Registry("DATASET")
DATASET_REGISTRY.__doc__ = """
Registry for dataset.

The registered object will be called with `obj(cfg, split)`.
The call should return a `torch.utils.data.Dataset` object.
"""


def build_dataset(dataset_name, cfg, split, is_features=False, title=None):
    """
    Build a dataset, defined by `dataset_name`.
    Args:
        dataset_name (str): the name of the dataset to be constructed.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        split (str): the split of the data loader. Options include `train`,
            `val`, and `test`.
        is_features (Bool): Whether to load features or videos
        title (str): Extra optional title to be printed
    Returns:
        Dataset: a constructed dataset specified by dataset_name.
    """
    return DATASET_REGISTRY.get(dataset_name)(cfg, split, is_features, title)
