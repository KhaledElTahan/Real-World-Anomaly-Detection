"""Information Utils"""

def get_dataset_features_name(cfg):
    """
    Creates dataset features name
    Args:
        cfg: The video model configuration file
    Returns:
        features_name (String): represents the dataset features name
            BackboneName_TransformationCode_InternalFramesBatchSize_NumberOfOutputSegments
    Example:
        get_dataset_features_name(cfg) => "Kinetics_c2_I3D_NLN_8x8_R50_BG-KNN_32x32"
    """
    return cfg.BACKBONE.NAME + "_" + cfg.TRANSFORM.CODE + "_" + \
        str(cfg.EXTRACT.FRAMES_BATCH_SIZE) + "x" + str(cfg.EXTRACT.NUMBER_OUTPUT_SEGMENTS)


def get_full_model_without_features(cfg):
    """
    Creates full model name without features name
    Args:
        cfg: The video model configuration file
    Returns:
        model_name_no_features (String): represents the model name without features
            ModelName_LossName_TrainingType_ReadingOrderCode_MOdelSignature
    Example:
        get_full_model_name(cfg) =>
            "SultaniBaseline_SultaniLoss_MIL_SHR_Baseline"
    """

    return cfg.MODEL.MODEL_NAME + "_" + cfg.MODEL.LOSS_FUNC + "_" + cfg.TRAIN.TYPE + \
        "_" + get_traing_reading_order_code (cfg) + "_" + cfg.MODEL.SIGN


def get_traing_reading_order_code(cfg):
    """
    Converts reading order into coded str
    Args:
        cfg: The video model configuration file
    Returns:
        reading_order_code (String): represents the reading order
    """
    reading_order_codes = {
        "Sequential": "SE",
        "Shuffle": "SH",
        "Shuffle with Replacement": "SHR",
        "Shuffle Pairs": "SHP",
    }

    assert cfg.TRAIN.DATA_READ_ORDER in reading_order_codes.keys()

    return reading_order_codes[cfg.TRAIN.DATA_READ_ORDER]


def get_full_model_name(cfg):
    """
    Creates full model name with features name
    Args:
        cfg: The video model configuration file
    Returns:
        full_model_name (String): represents the whole model name
            BackboneName_TransformationCode_InternalFramesBatchSize_
                NumberOfOutputSegments_ModelName_LossName_TrainingType_ReadingOrderCode_MOdelSignature
    Example:
        get_full_model_name(cfg) =>
            "Kinetics_c2_I3D_NLN_8x8_R50_BG-KNN_32x32_SultaniBaseline_SultaniLoss_MIL_SHR_Baseline"
    """
    return get_dataset_features_name(cfg) + "_" + get_full_model_without_features(cfg)
