"""A Module for the UCF Anomaly Dataset"""
import random
import torch
import torch.utils.data
from tabulate import tabulate

from src.datasets import decoder
from src.datasets import utils
from src.utils import pathutils
from src.datasets import video_container as container
from src.datasets.build import DATASET_REGISTRY
from src.datasets import transform_helper
from src.utils import funcutils
from src.models import backbone_helper
from src.datasets import transform

@DATASET_REGISTRY.register()
class UCFAnomalyDetection(torch.utils.data.Dataset):
    """
    UCFAnomalyDetection video loader.
    """

    def __init__(self, cfg, mode, num_retries=10):
        """
        Construct the UCF Anomaly Detection video loader with a given two txt files. 
        The format of the training file is:
            ```
            label_1/video_name_1
            label_2/video_name_2
            ...
            label_N/video_name_N
            ```
        The format of the testing file is:
            ```
            video_name_1 label_1 1st_anomaly_s_idx_1 1st_anomaly_e_idx_1 2nd_anomaly_s_idx_1 2nd_anomaly_e_idx_1
            video_name_2 label_2 1st_anomaly_s_idx_2 1st_anomaly_e_idx_2 2nd_anomaly_s_idx_2 2nd_anomaly_e_idx_2
            ...
            video_name_N label_N 1st_anomaly_s_idx_N 1st_anomaly_e_idx_N 2nd_anomaly_s_idx_N 2nd_anomaly_e_idx_N
            ```
            Notes:
                Each video might have zero, one, or two anomalies
                In case of:
                    Two Anomalies: video_name label 1st_anomaly_s_idx 1st_anomaly_e_idx 2nd_anomaly_s_idx 2nd_anomaly_e_idx
                    One Anomaly: video_name label 1st_anomaly_s_idx 1st_anomaly_e_idx -1 -1
                    Zero Anomalies: video_name Normal -1 -1 -1 -1
                
        URLs:
            https://www.crcv.ucf.edu/projects/real-world/
            https://visionlab.uncc.edu/download/summary/60-data/477-ucf-anomaly-detection-dataset
            https://www.dropbox.com/sh/75v5ehq4cdg5g5g/AABvnJSwZI7zXb8_myBA0CLHa?dl=0
        Args:
            cfg (CfgNode): configs.
            mode (string): Options includes `train`, or `test` mode.
            num_retries (int): number of retries.
        """
        # Only support train, and test mode.
        assert mode in [
            "train",
            "test",
        ], "Split '{}' not supported for UCF Anomaly Detection".format(mode)
        self.mode = mode
        self.cfg = cfg

        self._video_meta = {}
        self._num_retries = num_retries

        print("Constructing UCF Anomaly Detection {}...".format(mode))

        self._construct_loader()


    def _construct_loader(self):
        """
        Construct the video loader.
        """

        dataset_parent_directory = pathutils.get_datasets_path() / self.cfg.DATA.PATH_TO_DATA_DIR

        if self.mode == "train":
            path_to_file = dataset_parent_directory / "Anomaly_Train.txt"
        elif self.mode == "test":
            path_to_file = dataset_parent_directory / "Temporal_Anomaly_Annotation_for_Testing_Videos.txt"

        assert path_to_file.is_file(), "{} file not found".format(path_to_file)

        # Store all datasets combined
        self._path_to_videos = []
        self._labels = []
        self._temporal_annotations = []
        self._is_anomaly = []

        # Store anomaly and normal separately
        self._path_to_anomaly_videos = []
        self._anomaly_videos_labels = []
        self._anomaly_temporal_annotations = []

        self._path_to_normal_videos = []

        self._dataset_directory = pathutils.get_specific_dataset_path(self.cfg, self.mode, self.cfg.DATA.READ_FEATURES)

        labels_set = set()
        with path_to_file.open("r") as file_ptr:
            for index, line in enumerate(file_ptr.read().splitlines()):
                line = line.strip()
                
                if self.cfg.DATA.READ_FEATURES:
                    line = utils.video_name_to_features_name(line, self.cfg.DATA.EXT, self.cfg.EXTRACT.FEATURES_EXT)

                if self.mode == "train":
                    assert len(line.split(self.cfg.DATA.PATH_LABEL_SEPARATOR_TRAINING)) == 2

                    video_path = self._dataset_directory / line
                    video_label = line.split(self.cfg.DATA.PATH_LABEL_SEPARATOR_TRAINING)[0]
                    video_label = "Normal" if video_label == "Training_Normal_Videos_Anomaly" else video_label
                    temporal_annotation = (-1, -1, -1, -1)

                elif self.mode == "test":
                    assert len(line.split(self.cfg.DATA.PATH_LABEL_SEPARATOR_TESTING)) == 6

                    line_splitted = line.split(self.cfg.DATA.PATH_LABEL_SEPARATOR_TESTING)

                    label_in_path = line_splitted[1] if line_splitted[1] != "Normal" else "Testing_Normal_Videos_Anomaly"
                    video_path = self._dataset_directory / label_in_path / line_splitted[0]
                    video_label = line_splitted[1]
                    temporal_annotation = (
                        int(line_splitted[2]),
                        int(line_splitted[3]),
                        int(line_splitted[4]),
                        int(line_splitted[5]))

                labels_set.add(video_label)

                if not self._check_file_exists(video_path):
                    continue

                self._path_to_videos.append(video_path)
                self._labels.append(video_label)
                self._temporal_annotations.append(temporal_annotation)
                self._is_anomaly.append(video_label != "Normal")
                self._video_meta[index] = {}

                if video_label != "Normal":
                    self._path_to_anomaly_videos.append(video_path)
                    self._anomaly_videos_labels.append(video_label)
                    self._anomaly_temporal_annotations.append(temporal_annotation)
                else:
                    self._path_to_normal_videos.append(video_path)


        assert (
            len(self._path_to_videos) > 0
        ), "Failed to load UCF Anomaly Detection {} from {}".format(
            self.mode, path_to_file
        )

        assert(
            len(self._path_to_videos) == len(self._path_to_anomaly_videos) + len(self._path_to_normal_videos)
        )

        self.output_classes = sorted(list(labels_set))
        
        print(
            "Done Constructing UCF Anomaly Detection {} (size: {}) from {}".format(
                self.mode, len(self._path_to_videos), path_to_file
            )
        )

        self._print_dataset_stats()


    def _print_dataset_stats(self):
        """
        Prints a statistical summary of the dataset
        """
        dataset_type = "Videos" if not self.cfg.DATA.READ_FEATURES else "Features"
        dataset_action = "Extract Videos" if self.cfg.EXTRACT.ENABLE else "Video Model"
        current_files_output_classes = sorted(list(set(self._labels)))
        backbone_cfg = backbone_helper.get_backbone_merged_cfg(self.cfg)

        print("Dataset Summary:")

        headers = ["Attribute", "Value"]
        table = [
            ["Name", "UFC Anomaly Detection"],
            ["Mode", self.mode],
            ["Type", dataset_type],
            ["Action", dataset_action],
            ["Backbone", self.cfg.BACKBONE.NAME],
            ["SlowFast.Alpha", backbone_cfg.SLOWFAST.ALPHA]
                if backbone_cfg.MODEL.ARCH in backbone_cfg.MODEL.MULTI_PATHWAY_ARCH else None,
            ["Directory", self._dataset_directory],
            ["N. Output Classes", len(self.output_classes)],
            ["Output Classes", self.output_classes[:7]],
            ["Cont....", self.output_classes[7:]],
            ["N. Loaded Output Classes", len(current_files_output_classes)],
            ["Loaded Output Classes", current_files_output_classes[:7]],
            ["Cont....", current_files_output_classes[7:]] if len(current_files_output_classes) > 7 else None,
            ["Loading Files Type", self.cfg.DATA.USE_FILES],
            ["N. Loaded Videos", len(self._path_to_videos)],
            ["N. Loaded Anomaly Videos", len(self._path_to_anomaly_videos)],
            ["N. Loaded Normal Videos", len(self._path_to_normal_videos)],
        ]

        table = [x for x in table if x is not None]

        print(tabulate(table, headers, tablefmt="pretty", colalign=("center", "left")))
        print()


    def _check_file_exists(self, path):
        """
        Checks whether a file exists and takes an action according to self.cfg.DATA.USE_FILES
        Args:
            path (Path): Video of features file path
        Returns:
            True: if file exists or self.cfg.DATA.USE_FILES == "ignore"
            False: if file doesn't exist and self.cfg.DATA.USE_FILES == "available"
            Exception: if file doesn't exist and self.cfg.DATA.USE_FILES == "all"
        """
        if not path.exists():
            if self.cfg.DATA.USE_FILES == "all":
                raise RuntimeError("File {} doesn't exist, and cfg.DATA.USE_FILES is 'all'".format(path))
            elif self.cfg.DATA.USE_FILES == "available":
                return False
        return True


    def _decode_video(self, video_path):
        """"
        Load the video and decode it
        Args:
            video_path (Path): The absolute path of the video
        """
        video_container = None
        try:
            video_container = container.get_video_container(
                video_path,
                self.cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE,
                self.cfg.DATA.DECODING_BACKEND,
            )
        except Exception as e:
            print("Failed to load video from {} with error {}".format(video_path, e))

        # Return None if the current video was not able to access,
        # To allow a retrial from __getitem__
        if video_container is None:
            return None

        # Decode video. 
        frames = decoder.decode(video_container)

        # If decoding failed (wrong format, video is too short, and etc),
        # Return None to allow a retrial from __getitem__
        if frames is None:
            return None

        # Perform color normalization.
        frames = utils.tensor_normalize(
            frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
        )

        # T H W C -> C T H W.
        frames = frames.permute(3, 0, 1, 2)

        # Spatial Scaling
        if frames.shape[1] != self.cfg.DATA.SCALES[0] or frames.shape[2] != self.cfg.DATA.SCALES[1]:
            frames = transform.spatial_resize(frames, self.cfg.DATA.SCALES[0], self.cfg.DATA.SCALES[1])

        # Apply a list of transformations according to configurations
        frames = transform_helper.apply_transformations(frames, self.cfg)

        frames = utils.pack_pathway_output(self.cfg, frames)
        return frames


    def _get_frames_or_features(self, item_path):
        """
        Gets the frames of the item or the features based on self.cfg.DATA.READ_FEATURES
        Args:
            item_path (pathlib.Path): path for video or features file
        """
        if self.cfg.DATA.READ_FEATURES:
            return random.randint(0, 5)
        else:
            return self._decode_video(item_path)


    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            items (list(list(torch.Tensors))): the frames or features from the video.
                The dimension of torch.Tensor is `channel` x `num frames` x `height` x `width`.
                The first list is used for anomaly and normal distinction, i.e.
                    it will be on form of [normal_items, anomaly_items] or [items]
                The second list is used only incase if items is frames for pathway distinction
                    i.e. it will be on form of [slow_frames, slow_frames] or [frames]
                Thus, all possible formats for one or two videos are:
                    1) [normal_features, anomaly_features]
                    2) [features]
                    3) [normal_frames: [frames], anomaly_frames: [frames]]
                    4) [frames: [frames]]
                    5) [normal_frames: [slow_frames, fast_frames], anomaly_frames: [slow_frames, fast_frames]]
                    6) [frames: : [slow_frames, fast_frames]] -> In case of feature extraction
            label (list(int)): the label of the current one or two videos,
                on the form of [label] or ["Normal", label]
            annotation (list(tuples)): the annotations of current one or two videos, 
                on the form of [tuple] or [normal tuple, anomaly tuple] 
        """
        # 1) Extra work could be done here incase of different reading order.

        # If the video can not be decoded, 
        # repeatly find a random video replacement that can be decoded.
        for _ in range(self._num_retries):
            if self.mode == "train" and not self.cfg.EXTRACT.ENABLE:
                min_len = min(self._path_to_anomaly_videos, self._path_to_normal_videos)
                max_len = max(self._path_to_anomaly_videos, self._path_to_normal_videos)

                normal_idx = index % min_len
                anomaly_idx = index % max_len

                if self.cfg.TRAIN.SHIFT_INDEX:
                    anomaly_idx = (index + self.cfg.TRAIN.CURRENT_EPOCH) % max_len

                normal_items = self._get_frames_or_features(self._path_to_normal_videos[normal_idx])
                anomaly_items = self._get_frames_or_features(self._path_to_anomaly_videos[anomaly_idx])

                normal_label = "Normal"
                anomaly_label = self._anomaly_videos_labels[anomaly_idx]

                normal_annotations = (-1, -1, -1, -1)
                anomaly_annotations = self._anomaly_temporal_annotations[anomaly_idx]

                items = [normal_items, anomaly_items]
                labels = [normal_label, anomaly_label]
                annotations = [normal_annotations, anomaly_annotations]
            else: 
                items = [self._get_frames_or_features(self._path_to_videos[index])]
                labels = [self._labels[index]]
                annotations = [self._temporal_annotations[index]]

            if items is None or items[0] is None:
                index = random.randint(0, len(self._path_to_videos) - 1)
            else:
                break

        if items is None or items[0] is None:
            raise RuntimeError(
                "Failed to fetch video after {} retries.".format(
                    self._num_retries
                )
            )

        # Do I need to construct label as one hot vector or as an idx here?

        return items, labels, annotations


    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        # 1) Extra work could be done here incase of different reading order.

        if self.mode == "train" and not self.cfg.EXTRACT.ENABLE:
            # Make sure on __getitem__ to index using
            # idx % min(_path_to_anomaly_videos, _path_to_normal_videos)
            dataset_len = max(self._path_to_anomaly_videos, self._path_to_normal_videos)
        else:
            dataset_len = len(self._path_to_videos)

        return dataset_len
