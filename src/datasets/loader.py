"""Dataset loader."""

import random

from src.datasets import loader_helper
from src.datasets.build import build_dataset
from src.utils import funcutils


class DatasetLoader():
    """"
    Custom Loader for Datasets
    """

    def __init__(self, cfg, dataset_split, reading_features, reading_order, batch_size, drop_last=False):
        """
        Construct the dataset Loader
        Args:
            cfg (CfgNode): Video model configurations
            dataset_split (str): train or test
            reading_features (bool): True to read features, False for videos
            reading_order (str): "Sequential", "Shuffle", or "Shuffle with Replacement"
            batch_size (int): Batch size of get, see __getitem__ for more details
            drop_last (bool): if True, drops the last batch with size < batch_size
        """
        assert dataset_split in ["train", "test"]
        assert reading_order in ["Sequential", "Shuffle", "Shuffle with Replacement"]
        assert reading_features is True # Currently DatasetLoader supports this only

        self.cfg = cfg
        self.split = dataset_split
        self.is_features = reading_features
        self.reading_order = reading_order
        self.batch_size = batch_size
        self.drop_last = drop_last

        self._construct_dataset()
        self._construct_indices()


    def _construct_dataset(self):
        """
        Construct the dataset.
        """
        dataset_name = self.cfg.TRAIN.DATASET if self.split == 'train' else self.cfg.TEST.DATASET
        self.dataset = build_dataset(dataset_name, self.cfg, self.split, self.is_features)

        assert self.batch_size <= len(self.dataset)


    def _construct_indices(self):
        """"
        Construct the indices used in reading order
        """
        if self.split == "test":
            self.indices = list(range(len(self.dataset)))
        elif self.split == "train":
            self.indices_normal = list(range(self.dataset.len_normal()))
            self.indices_anomaly = list(range(self.dataset.len_anomalies()))

        if self.reading_order == "Shuffle":
            if self.split == "test":
                random.shuffle(self.indices)
            elif self.split == "train":
                random.shuffle(self.indices_normal)
                random.shuffle(self.indices_anomaly)


    def _get_batch_size(self, index):
        """
        Utility method to get batch size with respect to drop last
        Args:
            Index (int): Current batch index
        Returns:
            batch_size (int)
        """
        if self.drop_last or index + 1 < len(self):
            return self.batch_size
        else: ## index is last element now
            if self.split == "test":
                length = len(self.dataset)
            elif self.split == "train":
                length = min(self.dataset.len_normal(), self.dataset.len_anomalies())

            # An example to illustrate the idea:
            # length is 20
            # batch_size is 7
            # We need last 6
            # 20 % 7 = 20 - 2 * 7 = 6
            return length % self.batch_size


    @funcutils.debug(apply=False, sign=True, ret=True, sign_beautify=True, ret_beautify=True)
    def __getitem__(self, index):
        """
        if split == "train"
            Gets two batches of dataset with respect to batch_size
            One normal batch, and one anomaleous batch
        if split == "test"
            Gets one batch of dataset with respect to batch_size
        Args
            index (int): batch index
        Returns
            if split == "train"
                normal_batch (dict):
                anomaleous_batch (dict):
            if split == "test"
                batch (dict):
            each batch is of the format (dict):
            {
                features_batched: Tensor(Torch) features batched,
                labels: List(str) labels batched,
                one_hots: Tensor(Torch) one hot vectors batched,
                annotations: Tensor(Torch) segments annotations batched,
                paths: List(Path) features paths batched
            }
        """
        if index >= len(self):
            raise StopIteration

        current_batch_size = self._get_batch_size(index)

        def _get_indices(reading_order, batch_size, indices_list):
            if reading_order in ["Sequential", "Shuffle"]:
                dataset_index = index * self.batch_size # not current_batch_size
                indices = indices_list[dataset_index:dataset_index + batch_size]
            elif self.reading_order == "Shuffle with Replacement":
                random.shuffle(indices_list)
                indices = indices_list[0:batch_size]

            return indices


        def _indices_to_batch(indices, is_anomaly):
            results = []

            for idx in indices:
                if is_anomaly is None:
                    results.append(self.dataset[idx])
                elif is_anomaly in [True, False]:
                    results.append(self.dataset[idx, is_anomaly])

            return loader_helper.features_dataset_results_list_to_batch(results)


        if self.split == "train":
            indices_normal = _get_indices(self.reading_order, current_batch_size, self.indices_normal)
            indices_anomaly = _get_indices(self.reading_order, current_batch_size, self.indices_anomaly)
            return _indices_to_batch(indices_normal, False), _indices_to_batch(indices_anomaly, True)
        elif self.split == "test":
            indices = _get_indices(self.reading_order, current_batch_size, self.indices)
            return _indices_to_batch(indices, None)


    def __len__(self):
        """
        Returns the length of dataset loader with respect to batch size
        """
        if self.split == "test":
            length = len(self.dataset)
        elif self.split == "train":
            length = min(self.dataset.len_normal(), self.dataset.len_anomalies())

        if not self.drop_last and length % self.batch_size != 0:
            length += self.batch_size

        return length // self.batch_size
    