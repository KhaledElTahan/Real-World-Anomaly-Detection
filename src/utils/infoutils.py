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


def get_train_type(cfg):
    """
    Returns the model training type
    Args:
        cfg: The video model configuration file
    Returns:
        train_type (String): Representing the model training type
    """
    if cfg.TRAIN.TYPE in ['PL', 'PL-MIL']:
        return cfg.TRAIN.TYPE + "-" + cfg.TRAIN.PL_AUG_CODE
    else:
        return cfg.TRAIN.TYPE


def get_detailed_train_type(cfg):
    """
    Returns the information about the model training type
    Args:
        cfg: The video model configuration file
    Returns:
        detailed_train_type (String): Detailed information of the model training type
    """
    assert cfg.TRAIN.TYPE in ['MIL', 'PL', 'PL-MIL']

    detailed_train_type = ""
    if cfg.TRAIN.TYPE == "MIL":
        detailed_train_type = "Multiple Instance Learning"
    elif cfg.TRAIN.TYPE == "PL":
        detailed_train_type = \
            "Pseudo Labels - Base:{} Aug:{}".format(cfg.TRANSFORM.CODE, cfg.TRAIN.PL_AUG_CODE)
    elif cfg.TRAIN.TYPE == "PL-MIL":
        detailed_train_type = \
            "Pseudo Labels with Multiple Instance Learning - Base:{} Aug:{}".\
                format(cfg.TRANSFORM.CODE, cfg.TRAIN.PL_AUG_CODE)

    return detailed_train_type


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

    return cfg.MODEL.MODEL_NAME + "_" + cfg.MODEL.LOSS_FUNC + "_" + get_train_type(cfg) + \
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
        "All Pairs": "AP",
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
                NumberOfOutputSegments_ModelName_LossName_TrainingType_
                    ReadingOrderCode_MOdelSignature
    Example:
        get_full_model_name(cfg) =>
            "Kinetics_c2_I3D_NLN_8x8_R50_BG-KNN_32x32_SultaniBaseline_SultaniLoss_MIL_SHR_Baseline"
    """
    return get_dataset_features_name(cfg) + "_" + get_full_model_without_features(cfg)
