"""Configs."""
from fvcore.common.config import CfgNode

# ---------------------------------------------------------------------------- #
# Config definition
# ---------------------------------------------------------------------------- #
_C = CfgNode()


# -----------------------------------------------------------------------------
# Model options
# -----------------------------------------------------------------------------
_C.MODEL = CfgNode()

# Model architecture.
_C.MODEL.ARCH = "FC"

# Model name
_C.MODEL.MODEL_NAME = "SultaniBaseline"

# The number of classes to predict for the model.
_C.MODEL.NUM_CLASSES = 2

# Loss function.
_C.MODEL.LOSS_FUNC = "SultaniLoss"

# Pseudo Labels Loss, if TRAIN.TYPE in ['PL', 'PL-MIL']
_C.MODEL.PSEUDO_LOSS_FUNC = "PseudoLabelsLoss"

# Dropout rate.
_C.MODEL.DROPOUT_RATE = 0.6

# The std to initialize the fc layer(s).
_C.MODEL.FC_INIT_STD = 0.01

# Activation layer for the output head.
_C.MODEL.HEAD_ACT = "sigmoid"

# Model Signature
_C.MODEL.SIGN = "Baseline"

# Path to the checkpoint directory to load weights.
_C.MODEL.CHECKPOINTS_DIRECTORY = "model-checkpoints"

# -----------------------------------------------------------------------------
# Losses options
# -----------------------------------------------------------------------------
_C.LOSS = CfgNode()

# Sultani Smoothness Lambda
_C.LOSS.SL_SMOOTHNESS_LAMBDA = 8e-5

# Sultani Smoothness Lambda
_C.LOSS.SL_SPARISTY_LAMBDA = 8e-5

# ---------------------------------------------------------------------------- #
# Training options.
# ---------------------------------------------------------------------------- #
_C.TRAIN = CfgNode()

# If True Train the model, else skip training.
_C.TRAIN.ENABLE = True

# The type of the training
# "MIL", "PL", or "PL-MIL"
_C.TRAIN.TYPE = "MIL"

# if TRAIN.TYPE in ['PL', 'PL-MIL'] then choose Augmented features code
_C.TRAIN.PL_AUG_CODE = "BG-MOG2"

# if TRAIN.TYPE in ['PL', 'PL-MIL'] then choose pseudo label threshold [0.0, 1.0]
_C.TRAIN.PL_THRESHOLD = 0.6

# if TRAIN.TYPE in ['PL', 'PL-MIL'] then choose the source dataset of the normal pseudo label
# Either "ORG" for features without transformations or "AUG" for features with transformations
# SRC: The one we will choose least values indices from
# DST: The actuall dataset we will use to optimize model
_C.TRAIN.PL_NORMAL_LABEL_SRC = "AUG"
_C.TRAIN.PL_NORMAL_LABEL_DST = "AUG"

# if TRAIN.TYPE is 'PL-MIL
# List of intervals, each interval will use the PL_MIL_PERCENTAGE as a percentage for PL training
# Last Interval will always be repeated if all intervals have passed
_C.TRAIN.PL_MIL_INTERVALS = [10] + [5] * 5 + [5] * 5 + [10]

# if TRAIN.TYPE is 'PL-MIL
# List of percentages, each percentage represents the percentage of PL_MIL_INTERVALS interval
#   to train with MIL if PL_MIL_MILFIRST is True, or PL if False
# Must be of length TRAIN.PL_MIL_INTERVALS
_C.TRAIN.PL_MIL_PERCENTAGE = [1] + [0.6] * 5 + [0.4] * 5 + [0.5]

# if TRAIN.TYPE is 'PL-MIL
# If True, then each interval will begin with percentage of MIL Training
# If False, then each interval will begin with percentage of PL Training
_C.TRAIN.PL_MIL_MILFIRST = True

# Dataset for training.
_C.TRAIN.DATASET = "UCFAnomalyDetection"

# Available split for the dataset.
_C.TRAIN.DATASET_SPLITS = "train"

# Total mini-batch size.
_C.TRAIN.BATCH_SIZE = 64

# Evaluate model on test data every eval period epochs.
# Better leave it at 1 to avoid missing any rare good local minima
_C.TRAIN.EVAL_PERIOD = 1

# Enable evaluate model inside one training epoch
_C.TRAIN.ENABLE_EVAL_BATCH = False

# Evaluate model inside one training epoch, every set of batches
_C.TRAIN.EVAL_BATCH_PERIOD = 100

# Save model checkpoint every checkpoint period epochs.
_C.TRAIN.CHECKPOINT_PERIOD = 1

# Resume training from the latest checkpoint in the output directory.
_C.TRAIN.AUTO_RESUME = True

# Checkpoint types include `caffe2` or `pytorch`.
_C.TRAIN.CHECKPOINT_TYPE = "pytorch"

# Set it during the training
_C.TRAIN.CURRENT_EPOCH = 1

# Maximum number of training epochs
_C.TRAIN.MAX_EPOCH = 1000000

# If "Sequential", then read dataset sequentially until min(normal, anomaly)
# if "Shuffle", we will shuffle both normal and anomaly each epoch
# if "Shuffle with Replacement", we will shuffle both normal and anomaly each batch selection
# if "All Pairs", will create dataset of all pairs (Normal, Anomaly) ~ O(N^2)
# if "Shuffle Pairs", will create dataset of all pairs (Normal, Anomaly) ~ O(N^2) and
#   shuffle each epoch
_C.TRAIN.DATA_READ_ORDER = "Shuffle Pairs"


# ---------------------------------------------------------------------------- #
# Testing options
# ---------------------------------------------------------------------------- #
_C.TEST = CfgNode()

# If True test the model, else skip the testing.
_C.TEST.ENABLE = True

# Dataset for testing.
_C.TEST.DATASET = "UCFAnomalyDetection"

# Available split for the dataset.
_C.TEST.DATASET_SPLITS = "test"

# Total mini-batch size
_C.TEST.BATCH_SIZE = 128

# Path to the checkpoint to load the initial weight.
_C.TEST.CHECKPOINT_FILE_PATH = ""

# Checkpoint types include `caffe2` or `pytorch`.
_C.TEST.CHECKPOINT_TYPE = "pytorch"

# Path to saving prediction results file.
_C.TEST.SAVE_RESULTS_PATH = ""


# ---------------------------------------------------------------------------- #
# Inference options.
# ---------------------------------------------------------------------------- #
_C.INFER = CfgNode()

# If True evaluate the model, else skip evaluation.
_C.INFER.ENABLE = True

# Dataset.
_C.INFER.DATASET = ""

# Total mini-batch size.
_C.INFER.BATCH_SIZE = 64

# Path to the checkpoint to load the initial weight.
_C.INFER.CHECKPOINT_FILE_PATH = ""

# Checkpoint types include `caffe2` or `pytorch`.
_C.INFER.CHECKPOINT_TYPE = "pytorch"

# If True, perform inflation when loading checkpoint.
_C.INFER.CHECKPOINT_INFLATE = False


# ---------------------------------------------------------------------------- #
# Feature Extraction options
# ---------------------------------------------------------------------------- #
_C.EXTRACT = CfgNode()

# If True extract the data features, else skip the extraction.
_C.EXTRACT.ENABLE = True

# Dataset for extracting the features.
_C.EXTRACT.DATASET = "UCFAnomalyDetection"

# Available split for the dataset.
_C.EXTRACT.DATASET_SPLITS = ["train", "test"]

# Total mini-batch size
_C.EXTRACT.FRAMES_BATCH_SIZE = 32

# Number of frames batches to be evaluated by backbone
_C.EXTRACT.FRAMES_BATCHES_BATCH_SIZE = 4

# Number of segments of the output features
_C.EXTRACT.NUMBER_OUTPUT_SEGMENTS = 32

# Batch size of loaded videos
_C.EXTRACT.VIDEOS_BATCH_SIZE = 1 # at moment only 1 is supported

# The extension of results files.
_C.EXTRACT.FEATURES_EXT = "pt"

# If True, feature extraction will re extract the features even if the file exists
# If False, if features file exist, the extraction for this video will be skipped
_C.EXTRACT.FORCE_REWRITE = False

# -----------------------------------------------------------------------------
# Backbone options
# -----------------------------------------------------------------------------
_C.BACKBONE = CfgNode()

# General name to the backbone model
_C.BACKBONE.NAME = "Kinetics_c2_SLOWFAST_8x8_R50"

# Path to the configuration of the backbone model.
# Path root should be "configs"
_C.BACKBONE.CONFIG_FILE_PATH = "backbone-configs/Kinetics/c2/SLOWFAST_8x8_R50.yaml"

# Path to the checkpoint to load the initial weight.
# Path root should be "checkpoints"
_C.BACKBONE.CHECKPOINT_FILE_PATH = "backbone-checkpoints/Kinetics/c2/SLOWFAST_8x8_R50.pkl"

# If False -> Backbone will be part of the overall model
# If True -> Backbone will only be a feature extractor
_C.BACKBONE.FEATURE_EXTRACTION = True

# Use it in building the classifier
_C.BACKBONE.FEATURES_LENGTH = 2304

# If true fine tune the backbone model, else require no gradient.
_C.BACKBONE.TRAINABLE = False

# Use those attributes to inject them into the backbone cfg
# The backbone_cfg will set those attibutes with the mode cfg values
_C.BACKBONE.MERGE_CFG_LIST = [
    "NUM_GPUS",
    "BACKBONE.TRAINABLE",
]


# -----------------------------------------------------------------------------
# Data options
# -----------------------------------------------------------------------------
_C.DATA = CfgNode()

# The path to the data directory.
_C.DATA.PATH_TO_DATA_DIR = "UCF_Anomaly_Detection"

# If true, read from the extracted by "BACKBONE"
# If false, read from videos
_C.DATA.READ_FEATURES = False

# If true, preload all extracted videos to RAM
# If false, read extracted from disk
_C.DATA.FEATURES_PRELOAD = True

# If "all", an error will be created if all the dataset files aren't available
# If "available", will use only the available files, error will be created if empty
# If "ignore", will not check whether files exist or not
_C.DATA.USE_FILES = "available"

# If true, skip videos with size larger than DATA.MAX_VIDEO_SIZE
_C.DATA.SKIP_LARGE_VIDEOS = True

# Maximimum of Video size in bytes
# Only enforced when DATA.SKIP_LARGE_VIDEOS = True
_C.DATA.MAX_VIDEO_SIZE = 200 * 1024 * 1024

# The general separator used between path and label.
_C.DATA.PATH_LABEL_SEPARATOR = " "

# The training separator used between path and label.
_C.DATA.PATH_LABEL_SEPARATOR_TRAINING = "/"

# The validation separator used between path and label.
_C.DATA.PATH_LABEL_SEPARATOR_VALIDATION = " "

# The testing separator used between path and label.
_C.DATA.PATH_LABEL_SEPARATOR_TESTING = "  " # two spaces

# Video path prefix if any.
_C.DATA.PATH_PREFIX = ""

# The mean value of the video raw pixels across the R G B channels.
_C.DATA.MEAN = [0.45, 0.45, 0.45]

# Input frame channel dimensions.
_C.DATA.INPUT_CHANNEL_NUM = 3

# The std value of the video raw pixels across the R G B channels.
_C.DATA.STD = [0.225, 0.225, 0.225]

# The spatial scales.
_C.DATA.SCALES = [240, 320]

# Input videos may has different fps, convert it to the target video fps before
# frame sampling.
_C.DATA.TARGET_FPS = 30

# Decoding backend, options include `pyav` or `torchvision`
_C.DATA.DECODING_BACKEND = "pyav"

# If True, calculdate the map as metric.
_C.DATA.MULTI_LABEL = False

# Method to perform the ensemble, options include "sum" and "max".
_C.DATA.ENSEMBLE_METHOD = "sum"

# If True, revert the default input channel (RBG <-> BGR).
_C.DATA.REVERSE_INPUT_CHANNEL = False

# Videos Extension, used for working with Paths
_C.DATA.EXT = "mp4"


# ---------------------------------------------------------------------------- #
# Video decoding options
# ---------------------------------------------------------------------------- #
_C.VIDEO_DECODER = CfgNode()

# Decoding backend, options include `pyav` or `torchvision`
_C.VIDEO_DECODER.DECODING_BACKEND = "pyav"

# Enable multi thread decoding.
_C.VIDEO_DECODER.ENABLE_MULTI_THREAD_DECODE = False


# ---------------------------------------------------------------------------- #
# Common train/test data loader options
# ---------------------------------------------------------------------------- #
_C.DATA_LOADER = CfgNode()


# -----------------------------------------------------------------------------
# Data Transformations options
# -----------------------------------------------------------------------------
_C.TRANSFORM = CfgNode()

# A Code to distinguish the features extracted after applying the transformations
_C.TRANSFORM.CODE = "NONE"

# Control whether to subtract backround or not
_C.TRANSFORM.BG_SUBTRACTION_ENABLED = False

# IF Background subtraction is enabled
# Currently Supported Algorithms are KNN & MOG2
_C.TRANSFORM.BG_SUBTRACTION_ALGORITHM = "MOG2"

# ---------------------------------------------------------------------------- #
# Optimizer options
# ---------------------------------------------------------------------------- #
_C.OPTIMIZER = CfgNode()

# Optimizer Name
_C.OPTIMIZER.NAME = "Adagrad"

# Base learning rate.
_C.OPTIMIZER.BASE_LR = 0.01

# L2 regularization.
_C.OPTIMIZER.WEIGHT_DECAY = 8e-5

# Optimizer EPS
_C.OPTIMIZER.EPS = 1e-8


# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #

# Number of GPUs to use (applies to both training and testing).
_C.NUM_GPUS = 1

# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries.
_C.RNG_SEED = 1

# If True, log the model info.
_C.LOG_MODEL_INFO = True

# If True, run the run stats tool
_C.RUN_STATS_TOOL = False

# If True, run the run dev test tool
_C.RUN_DEV_TEST = False

# ---------------------------------------------------------------------------- #
# Benchmark options
# ---------------------------------------------------------------------------- #
_C.BENCHMARK = CfgNode()


# ---------------------------------------------------------------------------- #
# Visualizations options
# ---------------------------------------------------------------------------- #
_C.VISUALIZE = CfgNode()

# Plot line width
_C.VISUALIZE.PLIT_LINEWIDTH = 2

# ---------------------------------------------------------------------------- #
# Demo options
# ---------------------------------------------------------------------------- #
_C.DEMO = CfgNode()

# If True run the demo, otherwise skip
_C.DEMO.ENABLE = True


def _assert_and_infer_cfg(cfg):
    # TRAIN assertions.
    assert cfg.TRAIN.CHECKPOINT_TYPE in ["pytorch", "caffe2"]

    if cfg.NUM_GPUS:
        assert cfg.TRAIN.BATCH_SIZE % cfg.NUM_GPUS == 0

    return cfg


def get_cfg():
    """
    Get a copy of the default config.
    """
    return _assert_and_infer_cfg(_C.clone())
