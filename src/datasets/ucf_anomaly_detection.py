"""A Module for the UCF Anomaly Dataset"""
import random
import torch
import torch.utils.data

from src.datasets import decoder as decoder
from src.datasets import utils as utils
from src.utils import pathutils
from src.datasets import video_container as container
from src.datasets.build import DATASET_REGISTRY


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

        dataset_directory = pathutils.get_datasets_path() / self.cfg.DATA.PATH_TO_DATA_DIR

        if self.mode == "train":
            path_to_file = dataset_directory / "Anomaly_Train.txt"
        elif self.mode == "test":
            path_to_file = dataset_directory / "Temporal_Anomaly_Annotation_for_Testing_Videos.txt"

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

        if self.cfg.DATA.READ_FEATURES: # Read features
            features_name = self.cfg.BACKBONE.NAME + "_" + self.cfg.TRANSFORM.CODE
            dataset_directory = dataset_directory / "features" / features_name / self.mode
        else: # Read videos
            dataset_directory = dataset_directory / "videos" / self.mode

        with path_to_file.open("r") as file_ptr:
            for line in file_ptr.read().splitlines():
                line = line.strip()

                if self.mode == "train":
                    assert len(line.split(self.cfg.DATA.PATH_LABEL_SEPARATOR_TRAINING)) == 2

                    video_path = dataset_directory / line
                    video_label = line.split(self.cfg.DATA.PATH_LABEL_SEPARATOR_TRAINING)[0]
                    video_label = "Normal" if video_label == "Training_Normal_Videos_Anomaly" else video_label
                    temporal_annotation = (-1, -1, -1, -1)

                elif self.mode == "test":
                    assert len(line.split(self.cfg.DATA.PATH_LABEL_SEPARATOR_TESTING)) == 6

                    line_splitted = line.split(self.cfg.DATA.PATH_LABEL_SEPARATOR_TESTING)

                    video_path = dataset_directory / line_splitted[0]
                    video_label = line_splitted[1]
                    temporal_annotation = (
                        int(line_splitted[2]),
                        int(line_splitted[3]),
                        int(line_splitted[4]),
                        int(line_splitted[5]))

                self._path_to_videos.append(video_path)
                self._labels.append(video_label)
                self._temporal_annotations.append(temporal_annotation)
                self._is_anomaly.append(video_label != "Normal")

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

        self.output_classes = sorted(list(set(self._labels)))

        print(
            "DONE:: Constructing UCF Anomaly Detection {} (size: {}) from {}".format(
                self.mode, len(self._path_to_videos), path_to_file
            )
        )

        dataset_type = "Videos" if not self.cfg.DATA.READ_FEATURES else "Features"

        print()
        print("~::DATASET STATS::~")
        print("Name: UFC Anomaly Detection - Mode: {} - Type: {}".format(self.mode, dataset_type))
        print("Dataset actual directory: {}".format(dataset_directory))
        print("Number of output classes: {}".format(len(self.output_classes)))
        print("Output classes: ", end="")
        print(self.output_classes)
        print("Number of anomaly videos:", len(self._path_to_anomaly_videos))
        print("Number of normal videos:", len(self._path_to_normal_videos))



    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
        """
        short_cycle_idx = None
        # When short cycle is used, input index is a tupple.
        if isinstance(index, tuple):
            index, short_cycle_idx = index

        if self.mode in ["train", "val"]:
            # -1 indicates random sampling.
            temporal_sample_index = -1
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
            if short_cycle_idx in [0, 1]:
                crop_size = int(
                    round(
                        self.cfg.MULTIGRID.SHORT_CYCLE_FACTORS[short_cycle_idx]
                        * self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
            if self.cfg.MULTIGRID.DEFAULT_S > 0:
                # Decreasing the scale is equivalent to using a larger "span"
                # in a sampling grid.
                min_scale = int(
                    round(
                        float(min_scale)
                        * crop_size
                        / self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
        elif self.mode in ["test"]:
            temporal_sample_index = (
                self._spatial_temporal_idx[index]
                // self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            spatial_sample_index = (
                self._spatial_temporal_idx[index]
                % self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            min_scale, max_scale, crop_size = [self.cfg.DATA.TEST_CROP_SIZE] * 3
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale, crop_size}) == 1
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )
        sampling_rate = utils.get_random_sampling_rate(
            self.cfg.MULTIGRID.LONG_CYCLE_SAMPLING_RATE,
            self.cfg.DATA.SAMPLING_RATE,
        )
        # Try to decode and sample a clip from a video. If the video can not be
        # decoded, repeatly find a random video replacement that can be decoded.
        for _ in range(self._num_retries):
            video_container = None
            try:
                video_container = container.get_video_container(
                    self._path_to_videos[index],
                    self.cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE,
                    self.cfg.DATA.DECODING_BACKEND,
                )
            except Exception as e:
                print(
                    "Failed to load video from {} with error {}".format(
                        self._path_to_videos[index], e
                    )
                )
            # Select a random video if the current video was not able to access.
            if video_container is None:
                index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            # Decode video. Meta info is used to perform selective decoding.
            frames = decoder.decode(
                video_container,
                sampling_rate,
                self.cfg.DATA.NUM_FRAMES,
                temporal_sample_index,
                self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
                video_meta=self._video_meta[index],
                target_fps=self.cfg.DATA.TARGET_FPS,
                backend=self.cfg.DATA.DECODING_BACKEND,
                max_spatial_scale=max_scale,
            )

            # If decoding failed (wrong format, video is too short, and etc),
            # select another video.
            if frames is None:
                index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            # Perform color normalization.
            frames = utils.tensor_normalize(
                frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
            )
            # T H W C -> C T H W.
            frames = frames.permute(3, 0, 1, 2)
            # Perform data augmentation.
            frames = utils.spatial_sampling(
                frames,
                spatial_idx=spatial_sample_index,
                min_scale=min_scale,
                max_scale=max_scale,
                crop_size=crop_size,
                random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
                inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
            )

            label = self._labels[index]
            frames = utils.pack_pathway_output(self.cfg, frames)
            return frames, label, index, {}
        else:
            raise RuntimeError(
                "Failed to fetch video after {} retries.".format(
                    self._num_retries
                )
            )

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_videos)
