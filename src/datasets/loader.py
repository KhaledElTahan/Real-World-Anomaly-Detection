"""Dataset loader."""

import random
import copy

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
            reading_order (str):
                "Sequential", "Shuffle", "Shuffle with Replacement", "All Pairs", or "Shuffle Pairs"
            batch_size (int): Batch size of get, see __getitem__ for more details
            drop_last (bool): if True, drops the last batch with size < batch_size
        """
        assert reading_features is True, "Currently DatasetLoader supports features only"
        assert dataset_split in ["train", "test"]
        assert reading_order in \
            ["Sequential", "Shuffle", "Shuffle with Replacement", "All Pairs", "Shuffle Pairs"]

        if dataset_split == "test":
            assert reading_order in ["Sequential", "Shuffle", "Shuffle with Replacement"]

        self.cfg = cfg
        self.split = dataset_split
        self.is_features = reading_features
        self.reading_order = reading_order
        self.batch_size = batch_size
        self.drop_last = drop_last

        self._construct_dataset()
        self._construct_indices()
        self._construct_lengths()


    def _construct_dataset(self):
        """
        Construct the dataset.
        """
        title = "Base Train"\
            if self.split == 'train' and self.cfg.TRAIN.TYPE in ['PL', 'PL-MIL'] else None

        dataset_name = self.cfg.TRAIN.DATASET if self.split == 'train' else self.cfg.TEST.DATASET
        self.dataset = build_dataset(dataset_name, self.cfg, self.split, self.is_features, title)

        if self.split == 'train' and self.cfg.TRAIN.TYPE in ['PL', 'PL-MIL']:
            aug_weak_dataset_cfg = copy.deepcopy(self.cfg)
            aug_weak_dataset_cfg.TRANSFORM.CODE = aug_weak_dataset_cfg.TRAIN.PL_AUG_WEAK_CODE
            aug_strong_dataset_cfg = copy.deepcopy(self.cfg)
            aug_strong_dataset_cfg.TRANSFORM.CODE = aug_weak_dataset_cfg.TRAIN.PL_AUG_STRONG_CODE

            self.aug_weak_dataset = build_dataset(
                    dataset_name,
                    aug_weak_dataset_cfg,
                    self.split,
                    self.is_features,
                    "Weak Augmentation Train"
            )
            self.aug_strong_dataset = build_dataset(
                dataset_name,
                aug_strong_dataset_cfg,
                self.split,
                self.is_features,
                "Strong Augmentation Train"
            )

            assert self.dataset.len_normal() == self.aug_weak_dataset.len_normal()
            assert self.dataset.len_anomalies() == self.aug_weak_dataset.len_anomalies()
            assert self.dataset.len_normal() == self.aug_strong_dataset.len_normal()
            assert self.dataset.len_normal() == self.aug_strong_dataset.len_normal()


    def _construct_indices(self):
        """
        Construct the indices used in reading order
        """
        if self.split == "test":
            self.indices = list(range(len(self.dataset)))
        elif self.split == "train":
            if self.reading_order in ["All Pairs", "Shuffle Pairs"]:
                # Assume normal len = 3, and anomaly len = 4
                # normal_indices = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
                self.indices_normal = \
                    list(range(self.dataset.len_normal())) * self.dataset.len_anomalies()
                # anomaly_indices = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
                self.indices_anomaly = \
                    sorted(list(range(self.dataset.len_anomalies())) * self.dataset.len_normal())
            else:
                self.indices_normal = list(range(self.dataset.len_normal()))
                self.indices_anomaly = list(range(self.dataset.len_anomalies()))

        self._initial_shuffle()


    def _construct_lengths(self):
        """
        Calculate examples length and adapt batch size to it if necessary
        """
        if self.split == "test":
            self.examples_length = len(self.indices)
        elif self.split == "train":
            self.examples_length = min(len(self.indices_normal), len(self.indices_anomaly))

        if self.batch_size > self.examples_length:
            self.batch_size = self.examples_length

            if self.split == "test":
                self.cfg.TEST.BATCH_SIZE = self.examples_length
            elif self.split == "train":
                self.cfg.TRAIN.BATCH_SIZE = self.examples_length


    def _initial_shuffle(self):
        """Shuffle all lists before initial epoch"""
        if self.reading_order == "Shuffle":
            if self.split == "test":
                random.shuffle(self.indices)
            elif self.split == "train":
                random.shuffle(self.indices_normal)
                random.shuffle(self.indices_anomaly)
        elif self.reading_order == "Shuffle Pairs":
            # To keep the one to one mapping
            temp = list(zip(self.indices_normal, self.indices_anomaly))
            random.shuffle(temp)
            temp_normal, temp_anomaly = zip(*temp)
            self.indices_normal = list(temp_normal)
            self.indices_anomaly = list(temp_anomaly)


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
            if self.batch_size == self.examples_length:
                return self.batch_size

            # An example to illustrate the idea:
            # length is 20
            # batch_size is 7
            # We need last 6
            # 20 % 7 = 20 - 2 * 7 = 6
            return self.examples_length % self.batch_size


    @funcutils.debug(apply=False, sign=True, ret=True, sign_beautify=True, ret_beautify=True)
    def get_batch(self, dataset, index, allow_shuffle=True):
        """
        if split == "train"
            Gets two batches of dataset with respect to batch_size
            One normal batch, and one anomaleous batch
        if split == "test"
            Gets one batch of dataset with respect to batch_size
        Args
            dataset: Dataset to return the batch from
            index (int): batch index
            allow_shuffle (Bool): Allow shuffling for first index
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

        if allow_shuffle and index == 0:
            self._initial_shuffle()

        current_batch_size = self._get_batch_size(index)

        def _get_indices(reading_order, batch_size, indices_list):
            if reading_order in ["Sequential", "Shuffle", "All Pairs", "Shuffle Pairs"]:
                dataset_index = index * self.batch_size # not current_batch_size
                indices = indices_list[dataset_index:dataset_index + batch_size]
            elif reading_order == "Shuffle with Replacement":
                random.shuffle(indices_list)
                indices = indices_list[0:batch_size]

            return indices


        def _indices_to_batch(indices, is_anomaly):
            results = []

            for idx in indices:
                if is_anomaly is None:
                    results.append(dataset[idx])
                elif is_anomaly in [True, False]:
                    results.append(dataset[idx, is_anomaly])

            return loader_helper.features_dataset_results_list_to_batch(results)


        if self.split == "train":
            indices_normal = _get_indices(self.reading_order, current_batch_size, self.indices_normal)
            indices_anomaly = _get_indices(self.reading_order, current_batch_size, self.indices_anomaly)
            return _indices_to_batch(indices_normal, False), _indices_to_batch(indices_anomaly, True)
        elif self.split == "test":
            indices = _get_indices(self.reading_order, current_batch_size, self.indices)
            return _indices_to_batch(indices, None)


    @funcutils.debug(apply=False, sign=True, ret=True, sign_beautify=True, ret_beautify=True)
    def __getitem__(self, index):
        """
        if split == "train"
            if cfg.TRAIN.TYPE in ['PL', 'PL-MIL']
                Gets four batches from two datasets
                    Two from no transform dataset
                    Two from augmented dataset
                Each two is: one normal batch, one anomaly batch
            else
                Gets two batches of dataset with respect to batch_size
                One normal batch, and one anomaleous batch
        if split == "test"
            Gets one batch of dataset with respect to batch_size
        Args
            index (int): batch index
        Returns
            if split == "train"
                if cfg.TRAIN.TYPE in ['PL', 'PL-MIL']
                    org_normal_batch (dict), org_anomaleous_batch (dict),
                    aug_weak_normal_batch (dict), aug_weak_anomaleous_batch (dict),
                    aug_strong_normal_batch (dict), aug_strong_anomaleous_batch (dict)
                else
                    normal_batch (dict), anomaleous_batch (dict)
            if split == "test"
                batch (dict)

            each batch is of the format (dict):
            {
                features_batched: Tensor(Torch) features batched,
                labels: List(str) labels batched,
                one_hots: Tensor(Torch) one hot vectors batched,
                annotations: Tensor(Torch) segments annotations batched,
                paths: List(Path) features paths batched
            }
        """
        if self.split == 'train' and self.cfg.TRAIN.TYPE in ['PL', 'PL-MIL']:
            org_normal_batch, org_anomaleous_batch = self.get_batch(self.dataset, index)
            aug_weak_normal_batch, aug_weak_anomaleous_batch =\
                self.get_batch(self.aug_weak_dataset, index, False)
            aug_strong_normal_batch, aug_strong_anomaleous_batch =\
                self.get_batch(self.aug_strong_dataset, index, False)
            return org_normal_batch, org_anomaleous_batch,\
                aug_weak_normal_batch, aug_weak_anomaleous_batch,\
                aug_strong_normal_batch, aug_strong_anomaleous_batch
        else:
            return self.get_batch(self.dataset, index)


    def __len__(self):
        """
        Returns the length of dataset loader with respect to batch size
        """
        length = self.examples_length

        if not self.drop_last and length % self.batch_size != 0:
            length += self.batch_size

        return length // self.batch_size
