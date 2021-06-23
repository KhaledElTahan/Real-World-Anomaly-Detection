"""Configs."""
from fvcore.common.config import CfgNode

# ---------------------------------------------------------------------------- #
# Config definition
# ---------------------------------------------------------------------------- #
_C = CfgNode()

# ---------------------------------------------------------------------------- #
# Evaluation options.
# ---------------------------------------------------------------------------- #
_C.EVAL = CfgNode()

# If True evaluate the model, else skip evaluation.
_C.EVAL.ENABLE = True

# Dataset.
_C.EVAL.DATASET = ""

# Total mini-batch size.
_C.EVAL.BATCH_SIZE = 64

# Path to the checkpoint to load the initial weight.
_C.EVAL.CHECKPOINT_FILE_PATH = ""

# Checkpoint types include `caffe2` or `pytorch`.
_C.EVAL.CHECKPOINT_TYPE = "pytorch"

# If True, perform inflation when loading checkpoint.
_C.EVAL.CHECKPOINT_INFLATE = False

# ---------------------------------------------------------------------------- #
# Training options.
# ---------------------------------------------------------------------------- #
_C.TRAIN = CfgNode()

# If True Train the model, else skip training.
_C.TRAIN.ENABLE = True

# Dataset.
_C.TRAIN.DATASET = ""

# Total mini-batch size.
_C.TRAIN.BATCH_SIZE = 64

# Evaluate model on test data every eval period epochs.
_C.TRAIN.EVAL_PERIOD = 1

# Save model checkpoint every checkpoint period epochs.
_C.TRAIN.CHECKPOINT_PERIOD = 1

# Resume training from the latest checkpoint in the output directory.
_C.TRAIN.AUTO_RESUME = True

# Path to the checkpoint to load the initial weight.
_C.TRAIN.CHECKPOINT_FILE_PATH = ""

# Checkpoint types include `caffe2` or `pytorch`.
_C.TRAIN.CHECKPOINT_TYPE = "pytorch"

# If True, perform inflation when loading checkpoint.
_C.TRAIN.CHECKPOINT_INFLATE = False

# Set it during the training, used in index shifting
_C.TRAIN.CURRENT_EPOCH = 0

# If true, each epoch we shift the mapping between one normal and one anomaly video
# If false, training for all epochs happens with the same pair of (normal, anomaly)
_C.TRAIN.SHIFT_INDEX = True

# ---------------------------------------------------------------------------- #
# Testing options
# ---------------------------------------------------------------------------- #
_C.TEST = CfgNode()

# If True test the model, else skip the testing.
_C.TEST.ENABLE = True

# Dataset for testing.
_C.TEST.DATASET = "kinetics"

# Total mini-batch size
_C.TEST.BATCH_SIZE = 8

# Path to the checkpoint to load the initial weight.
_C.TEST.CHECKPOINT_FILE_PATH = ""

# Number of clips to sample from a video uniformly for aggregating the
# prediction results.
_C.TEST.NUM_ENSEMBLE_VIEWS = 10

# Number of crops to sample from a frame spatially for aggregating the
# prediction results.
_C.TEST.NUM_SPATIAL_CROPS = 3

# Checkpoint types include `caffe2` or `pytorch`.
_C.TEST.CHECKPOINT_TYPE = "pytorch"

# Path to saving prediction results file.
_C.TEST.SAVE_RESULTS_PATH = ""

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

# If true fine tune the mode, else require no gradient.
_C.BACKBONE.TRAINABLE = False

# Use those attributes to inject them into the backbone cfg
# The backbone_cfg will set those attibutes with the mode cfg values
_C.BACKBONE.MERGE_CFG_LIST = [
    "NUM_GPUS",
    "BACKBONE.TRAINABLE",
]

# -----------------------------------------------------------------------------
# Model options
# -----------------------------------------------------------------------------
_C.MODEL = CfgNode()

# Model architecture.
_C.MODEL.ARCH = ""

# Model name
_C.MODEL.MODEL_NAME = ""

# The number of classes to predict for the model.
_C.MODEL.NUM_CLASSES = 0

# Loss function.
_C.MODEL.LOSS_FUNC = ""

# Dropout rate.
_C.MODEL.DROPOUT_RATE = 0.5

# The std to initialize the fc layer(s).
_C.MODEL.FC_INIT_STD = 0.01

# Activation layer for the output head.
_C.MODEL.HEAD_ACT = ""

# -----------------------------------------------------------------------------
# Data options
# -----------------------------------------------------------------------------
_C.DATA = CfgNode()

# The path to the data directory.
_C.DATA.PATH_TO_DATA_DIR = "UCF_Anomaly_Detection"

# If true, read from the extracted by "BACKBONE"
# If false, read from videos
_C.DATA.READ_FEATURES = False

# If "all", an error will be created if all the dataset files aren't available
# If "available", will use only the available files, error will be created if empty
# If "ignore", will not check whether files exist or not
_C.DATA.USE_FILES = "available"

# If true, skip videos with size larger than DATA.MAX_VIDEO_SIZE
_C.DATA.SKIP_LARGE_VIDEOS = True

# Maximimum of Video size in bytes
# Only enforced when DATA.SKIP_LARGE_VIDEOS = True
_C.DATA.MAX_VIDEO_SIZE = 500 * 1024 * 1024

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


# -----------------------------------------------------------------------------
# Data Transformations options
# -----------------------------------------------------------------------------
_C.TRANSFORM = CfgNode()

# A Code to distinguish the features extracted after applying the transformations
_C.TRANSFORM.CODE = "NONE"

# Control whether to subtract backround or not
_C.TRANSFORM.BG_SUBTRACTION_ENABLED = True

# IF Background subtraction is enabled
# Currently Supported Algorithms are KNN & MOG2
_C.TRANSFORM.BG_SUBTRACTION_ALGORITHM = "MOG2"

# ---------------------------------------------------------------------------- #
# Optimizer options
# ---------------------------------------------------------------------------- #
_C.SOLVER = CfgNode()

# Base learning rate.
_C.SOLVER.BASE_LR = 0.1

# Learning rate policy (see utils/lr_policy.py for options and examples).
_C.SOLVER.LR_POLICY = "cosine"

# Exponential decay factor.
_C.SOLVER.GAMMA = 0.1

# Step size for 'exp' and 'cos' policies (in epochs).
_C.SOLVER.STEP_SIZE = 1

# Steps for 'steps_' policies (in epochs).
_C.SOLVER.STEPS = []

# Learning rates for 'steps_' policies.
_C.SOLVER.LRS = []

# Maximal number of epochs.
_C.SOLVER.MAX_EPOCH = 300

# Momentum.
_C.SOLVER.MOMENTUM = 0.9

# Momentum dampening.
_C.SOLVER.DAMPENING = 0.0

# Nesterov momentum.
_C.SOLVER.NESTEROV = True

# L2 regularization.
_C.SOLVER.WEIGHT_DECAY = 1e-4

# Start the warm up from SOLVER.BASE_LR * SOLVER.WARMUP_FACTOR.
_C.SOLVER.WARMUP_FACTOR = 0.1

# Gradually warm up the SOLVER.BASE_LR over this number of epochs.
_C.SOLVER.WARMUP_EPOCHS = 0.0

# The start learning rate of the warm up.
_C.SOLVER.WARMUP_START_LR = 0.01

# Optimization method.
_C.SOLVER.OPTIMIZING_METHOD = "sgd"


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


# ---------------------------------------------------------------------------- #
# Benchmark options
# ---------------------------------------------------------------------------- #
_C.BENCHMARK = CfgNode()


# ---------------------------------------------------------------------------- #
# Common train/test data loader options
# ---------------------------------------------------------------------------- #
_C.DATA_LOADER = CfgNode()

# Number of data loader workers per training process.
_C.DATA_LOADER.NUM_WORKERS = 8

# Load data to pinned host memory.
_C.DATA_LOADER.PIN_MEMORY = True

# Enable multi thread decoding.
_C.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE = False


# ---------------------------------------------------------------------------- #
# Detection options.
# ---------------------------------------------------------------------------- #
_C.DETECTION = CfgNode()


# ---------------------------------------------------------------------------- #
# Multigrid training options
# See https://arxiv.org/abs/1912.00998 for details about multigrid training.
# ---------------------------------------------------------------------------- #
_C.MULTIGRID = CfgNode()

# ---------------------------------------------------------------------------- #
# Tensorboard Visualization Options
# ---------------------------------------------------------------------------- #
_C.TENSORBOARD = CfgNode()


# ---------------------------------------------------------------------------- #
# Demo options
# ---------------------------------------------------------------------------- #
_C.DEMO = CfgNode()


def _assert_and_infer_cfg(cfg):
    # TRAIN assertions.
    assert cfg.TRAIN.CHECKPOINT_TYPE in ["pytorch", "caffe2"]

    if cfg.NUM_GPUS:
        assert cfg.TRAIN.BATCH_SIZE % cfg.NUM_GPUS == 0

    # TEST assertions.
    assert cfg.TEST.CHECKPOINT_TYPE in ["pytorch", "caffe2"]
    assert cfg.TEST.BATCH_SIZE % cfg.NUM_GPUS == 0
    assert cfg.TEST.NUM_SPATIAL_CROPS == 3

    return cfg


def get_cfg():
    """
    Get a copy of the default config.
    """
    return _assert_and_infer_cfg(_C.clone())
