"""Dataset loader."""

import random

from torch.utils.data import dataset
from src.datasets import loader_helper
from src.datasets.build import build_dataset
from src.utils import funcutils


class DatasetLoader():
    """"
    Custom Loader for Datasets
    """

    def __init__(self, cfg, dataset_split, reading_features, reading_order, batch_size):
        """
        Construct the dataset Loader
        Args:
            cfg (CfgNode): Video model configurations
            dataset_split (str): train or test
            reading_features (bool): True to read features, False for videos
            reading_order (str): "Sequential", "Shuffle", or "Shuffle with Replacement"
            batch_size (int): Batch size of anomalies and normal items (videos or features)
                i.e. will return batch_size x 2 items
        """
        assert dataset_split in ["train", "test"]
        assert reading_order in ["Sequential", "Shuffle", "Shuffle with Replacement"]
        assert reading_features is True # Currently DatasetLoader supports this only

        self.cfg = cfg
        self.split = dataset_split
        self.is_features = reading_features
        self.reading_order = reading_order
        self.batch_size = batch_size

        self._construct_dataset()
        self._construct_indices()


    def _construct_dataset(self):
        """
        Construct the dataset.
        """
        dataset_name = self.cfg.TRAIN.DATASET if self.split == 'train' else self.cfg.TEST.DATASET
        self.dataset = build_dataset(dataset_name, self.cfg, self.split, self.is_features)

        assert self.batch_size <= len(self)


    def _construct_indices(self):
        """"
        Construct the indices used in reading order
        """
        self.indices = list(range(len(self) * self.batch_size))

        if self.reading_order == "Shuffle":
            random.shuffle(self.indices)


    @funcutils.debug(apply=True, sign=True, ret=True, sign_beautify=True, ret_beautify=True)
    def __getitem__(self, index):
        """
        Gets two batches of dataset with respect to batch_size
        One normal batch, and one anomaleous batch
        Args
            index (int): batch index
        Returns
            normal_batch (dict):
                see loader_helper.features_dataset_results_list_to_batch for more details
            anomaleous_batch (dict):
                see loader_helper.features_dataset_results_list_to_batch for more details
        """
        if index >= len(self):
            raise StopIteration

        if self.reading_order in ["Sequential", "Shuffle"]:
            dataset_index = index * self.batch_size
            indices = self.indices[dataset_index:dataset_index + self.batch_size]
        elif self.reading_order == "Shuffle with Replacement":
            random.shuffle(self.indices)
            indices = self.indices[0:self.batch_size]

        normal_results = []
        anomaleous_results = []

        for idx in indices:
            normal_results.append(self.dataset[idx, False])
            anomaleous_results.append(self.dataset[idx, True])

        normal_batch = loader_helper.features_dataset_results_list_to_batch(normal_results)
        anomaleous_batch = loader_helper.features_dataset_results_list_to_batch(anomaleous_results)

        return normal_batch, anomaleous_batch


    def __len__(self):
        """
        Returns the length of dataset loader with respect to batch size
        """
        return min(self.dataset.len_normal(), self.dataset.len_anomalies()) // self.batch_size
