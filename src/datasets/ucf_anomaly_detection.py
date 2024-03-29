"""A Module for the UCF Anomaly Dataset"""
import random
import torch
import torch.utils.data
import numpy as np
from tabulate import tabulate

from src.datasets import decoder
from src.datasets import utils
from src.utils import infoutils, pathutils
from src.datasets import video_container as container
from src.datasets.build import DATASET_REGISTRY
from src.datasets import transform_helper
from src.utils import funcutils
from src.utils import misc

@DATASET_REGISTRY.register()
class UCFAnomalyDetection(torch.utils.data.Dataset):
    """
    UCFAnomalyDetection video loader.
    """

    def __init__(self, cfg, mode, is_features=False, title=None):
        """
        Construct the UCF Anomaly Detection video loader with a given two txt files.
        The format of the training file is:
            ```
            label_1/video_name_1/video_size_1
            label_2/video_name_2/video_size_2
            ...
            label_N/video_name_N/video_size_N
            ```
        The format of the testing file is:
            ```
            video_name_1 label_1 1st_anomaly_s_idx_1 1st_anomaly_e_idx_1 2nd_anomaly_s_idx_1 2nd_anomaly_e_idx_1 video_size_1
            video_name_2 label_2 1st_anomaly_s_idx_2 1st_anomaly_e_idx_2 2nd_anomaly_s_idx_2 2nd_anomaly_e_idx_2 video_size_2
            ...
            video_name_N label_N 1st_anomaly_s_idx_N 1st_anomaly_e_idx_N 2nd_anomaly_s_idx_N 2nd_anomaly_e_idx_N video_size_N
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
            is_features (Bool): Whether to load features or videos
            title (str): Extra optional title to be printed
        """
        # Only support train, and test mode.
        assert mode in [
            "train",
            "test",
        ], "Split '{}' not supported for UCF Anomaly Detection".format(mode)
        self.mode = mode
        self.cfg = cfg
        self.is_features = is_features

        self._video_meta = {}

        if title is None:
            print("Constructing UCF Anomaly Detection ({}) - ".format(mode), end="")
        else:
            print("Constructing UCF Anomaly Detection ({}) - ".format(title), end="")

        self._construct_dataset()


    def _construct_dataset(self):
        """
        Construct the dataset.
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
        self._anomaly_index_to_general_index = []

        self._path_to_normal_videos = []
        self._normal_index_to_general_index = []

        self._dataset_directory = pathutils.get_specific_dataset_path(self.cfg, self.is_features)

        labels_set = set()
        _general_index = 0
        with path_to_file.open("r") as file_ptr:
            for index, line in enumerate(file_ptr.read().splitlines()):
                line = line.strip()
                
                if self.is_features:
                    line = pathutils.change_extension(line, self.cfg.DATA.EXT, self.cfg.EXTRACT.FEATURES_EXT)

                if self.mode == "train":
                    line_splitted = line.split(self.cfg.DATA.PATH_LABEL_SEPARATOR_TRAINING)

                    assert len(line_splitted) == 3

                    video_path = self._dataset_directory / line_splitted[0] / line_splitted[1]
                    video_label = line_splitted[0]
                    video_label = "Normal" if video_label == "Training_Normal_Videos_Anomaly" else video_label
                    temporal_annotation = (-1, -1, -1, -1)
                    video_size = int(line_splitted[-1])

                elif self.mode == "test":
                    line_splitted = line.split(self.cfg.DATA.PATH_LABEL_SEPARATOR_TESTING)

                    assert len(line_splitted) == 7

                    label_in_path = line_splitted[1] if line_splitted[1] != "Normal" else "Testing_Normal_Videos_Anomaly"
                    video_path = self._dataset_directory / label_in_path / line_splitted[0]
                    video_label = line_splitted[1]
                    temporal_annotation = (
                        int(line_splitted[2]),
                        int(line_splitted[3]),
                        int(line_splitted[4]),
                        int(line_splitted[5]))
                    video_size = int(line_splitted[-1])

                labels_set.add(video_label)

                if not self._check_file_exists(video_path):
                    continue

                if self.cfg.DATA.SKIP_LARGE_VIDEOS and video_size > self.cfg.DATA.MAX_VIDEO_SIZE:
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
                    self._anomaly_index_to_general_index.append(_general_index)
                else:
                    self._path_to_normal_videos.append(video_path)
                    self._normal_index_to_general_index.append(_general_index)
                    
                _general_index += 1

        assert (
            len(self._path_to_videos) > 0
        ), "\nFailed to load UCF Anomaly Detection {} from {}".format(
            self.mode, path_to_file
        )

        assert(
            len(self._path_to_videos) == len(self._path_to_anomaly_videos) + len(self._path_to_normal_videos)
        )

        self.output_classes = sorted(list(labels_set))
        
        print("SUCCESS - ", end="")

        if self.is_features and self.cfg.DATA.FEATURES_PRELOAD:
            self._preload_features()

        self._print_dataset_stats()


    def _print_dataset_stats(self):
        """
        Prints a statistical summary of the dataset
        """
        dataset_type = "Videos" if not self.is_features else "Features"
        current_files_output_classes = sorted(list(set(self._labels)))
        max_size = "Unlimited"

        if self.cfg.DATA.SKIP_LARGE_VIDEOS:
            max_size = misc.sizeof_fmt(self.cfg.DATA.MAX_VIDEO_SIZE)

        print("Dataset Summary:")

        headers = ["Attribute", "Value"]
        table = [
            ["Name", "UFC Anomaly Detection"],
            ["Mode", self.mode],
            ["Type", dataset_type],
            ["Features Name", infoutils.get_dataset_features_name(self.cfg)]
                if self.is_features else None,
            ["Transformation Code", self.cfg.TRANSFORM.CODE],
            ["Preload Features", self.cfg.DATA.FEATURES_PRELOAD] if self.is_features else None,
            ["Directory", self._dataset_directory],
            ["N. Output Classes", len(self.output_classes)],
            ["Output Classes", self.output_classes[:7]],
            ["Cont....", self.output_classes[7:]],
            ["N. Loaded Output Classes", len(current_files_output_classes)],
            ["Loaded Output Classes", current_files_output_classes[:7]],
            ["Cont....", current_files_output_classes[7:]] if len(current_files_output_classes) > 7 else None,
            ["Loading Files Type", self.cfg.DATA.USE_FILES],
            ["Maximum Video Size", max_size],
            ["N. Loaded Videos", len(self._path_to_videos)],
            ["N. Loaded Anomaly Videos", len(self._path_to_anomaly_videos)],
            ["N. Loaded Normal Videos", len(self._path_to_normal_videos)],
        ]

        table = [x for x in table if x is not None]

        print(tabulate(table, headers, tablefmt="pretty", colalign=("center", "left")))
        print()


    def _preload_features(self):
        """
        Preload all features to RAM to improve performance
        """
        self._features_segments_list = []
        self._is_anomaly_segment_list = []

        for features_path in self._path_to_videos:
            loaded = torch.load(features_path)

            features_segments = loaded['features_segments']
            is_anomaly_segment = loaded['is_anomaly_segment']

            self._features_segments_list.append(features_segments)
            self._is_anomaly_segment_list.append(is_anomaly_segment)

            assert features_segments.shape[1] == self.cfg.BACKBONE.FEATURES_LENGTH, \
                "The backbone configuration features length doesn't match with extracted data\n" + \
                "Extracted data length {} - Backbone configurations features length {}". \
                format(features_segments.shape[1], self.cfg.BACKBONE.FEATURES_LENGTH)


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


    @funcutils.debug(apply=False, sign=True, ret=True, sign_beautify=True, ret_beautify=True)
    @funcutils.force_garbage_collection(before=True, after=True)
    def _get_video(self, video_path):
        """"
        Loads the video, decodes it, applies video transformations then packs the output frames
        for backbone pathway input
        Args:
            video_path (Path): The absolute path of the video
        Returns:
            frames (list(Torch.Tensor)): List of frames tensors, each tensor represents transformed
                video frames for backbone input pathway
        """
        video_container = None
        try:
            assert video_path.exists()

            video_container = container.get_video_container(
                video_path,
                self.cfg.VIDEO_DECODER.ENABLE_MULTI_THREAD_DECODE,
                self.cfg.VIDEO_DECODER.DECODING_BACKEND,
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

        # list of np frames
        frames = [frame.to_rgb().to_ndarray() for frame in frames]
        frames = transform_helper.apply_transformations_list_np_frames(self.cfg, frames)

        # np tensor
        frames = np.stack(frames)
        frames = transform_helper.apply_transformations_np_frames(self.cfg, frames) # Type=uint8

        # torch tensor
        frames = torch.as_tensor(frames) # Type=torch.uint8
        frames = transform_helper.apply_transformations_THWC_torch_frames(self.cfg, frames) # Type=torch.float32

        # T H W C -> C T H W.
        frames = frames.permute(3, 0, 1, 2)
        frames = transform_helper.apply_transformations_CTHW_torch_frames(self.cfg, frames)

        # Prepare data for multipathway backbone models
        frames = utils.pack_pathway_output(self.cfg, frames)
        return frames

    
    @funcutils.debug(apply=False, sign=True, ret=True, sign_beautify=True, ret_beautify=True)
    def _get_features(self, general_index, features_path):
        """"
        Loads the features produced by the backbone, if preloading is true, it reads it from RAM
            otherwise it reads it from disk
        Args:
            general_index (int): Index of item in general list, used for preloading
            features_path (Path): The absolute path of the features
        Returns:
            features_segments (Torch.Tensor): feature representation of video segments
            is_anomaly_segment (Torch.Tensor): boolean vector, each cell represents whether
                the coresponding video segment is anomaly (True) or normal (False)
        """
        if self.cfg.DATA.FEATURES_PRELOAD:
            features_segments = self._features_segments_list[general_index]
            is_anomaly_segment = self._is_anomaly_segment_list[general_index]
        else:
            loaded = torch.load(features_path)

            features_segments = loaded['features_segments']
            is_anomaly_segment = loaded['is_anomaly_segment']

            assert features_segments.shape[1] == self.cfg.BACKBONE.FEATURES_LENGTH, \
                "The backbone configuration features length doesn't match with extracted data\n" + \
                "Extracted data length {} - Backbone configurations features length {}". \
                format(features_segments.shape[1], self.cfg.BACKBONE.FEATURES_LENGTH)

        return features_segments, is_anomaly_segment


    def _get_frames_or_features(self, general_index, item_path):
        """
        Gets the frames of the item or the features based on self.is_features
        Args:
            general_index (int): Index of item in general list, used for preloading
            item_path (pathlib.Path): path for video or features file
        """
        if self.is_features:
            return self._get_features(general_index, item_path)
        else:
            return self._get_video(item_path)


    @funcutils.debug(apply=False, sign=True, ret=True, sign_beautify=True, ret_beautify=True)
    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int or tupe(int, bool)):
                index (int): the video index provided by the pytorch sampler.
                anomaly (bool):
                    if None: retrieve all dataset
                    if True: Retrieve only anomalies
                    if False: Retrieve only notmal videos
        Returns:
            item (Torch.Tensor): frames or features depending on self.is_features
            label (Str): label of the dataset item
            one_hot (Torch.Tensor): label encoded as one hot vector
            annotation (Tuple or Tensor): tuple representing anomaleous frames or tensor of segments
                depending on self.is_features
            item_path (Path): video or features file path depending on self.is_features
        """
        assert isinstance(index, (int, tuple))

        if isinstance(index, tuple):
            index, anomaly = index
        else:
            anomaly = None

        if anomaly is None:
            paths_list = self._path_to_videos
            labels_list = self._labels
            temporal_annotations_list = self._temporal_annotations
            general_index = index
        elif anomaly is True:
            paths_list = self._path_to_anomaly_videos
            labels_list = self._anomaly_videos_labels
            temporal_annotations_list = self._anomaly_temporal_annotations
            general_index = self._anomaly_index_to_general_index[index]
        elif anomaly is False:
            paths_list = self._path_to_normal_videos
            labels_list = ["Normal"] * self.len_normal()
            temporal_annotations_list = [(-1, -1, -1, -1)] * self.len_normal()
            general_index = self._normal_index_to_general_index[index]

        skip_reading = False

        if not self.is_features:
            features_path = pathutils.video_path_to_features_path(self.cfg, paths_list[index])

            skip_reading = self.cfg.EXTRACT.ENABLE and \
                not self.cfg.EXTRACT.FORCE_REWRITE and \
                features_path.exists()

        if skip_reading:
            item = None
            label = None
            one_hot = None
            annotation = None
        else:
            item = self._get_frames_or_features(general_index, paths_list[index])
            label = labels_list[index]
            one_hot = utils.label_to_one_hot(label, self.output_classes)

            if self.is_features:
                item, annotation = item
            else:
                annotation = temporal_annotations_list[index]

        if not skip_reading and item is None:
            raise RuntimeError("Failed to fetch video.")

        return item, label, one_hot, annotation, paths_list[index]


    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_videos)


    def len_anomalies(self):
        """
        Returns:
            (int): the number of anomaly videos in the dataset.
        """
        return len(self._path_to_anomaly_videos)


    def len_normal(self):
        """
        Returns:
            (int): the number of normal videos in the dataset.
        """
        return len(self._path_to_normal_videos)

