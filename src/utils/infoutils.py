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


def get_full_model_name(cfg):
    """
    Creates dataset features name
    Args:
        cfg: The video model configuration file
    Returns:
        full_model_name (String): represents the whole model name
            BackboneName_TransformationCode_InternalFramesBatchSize_
                NumberOfOutputSegments_ModelName_LossName_TrainingType_MOdelSignature
    Example:
        get_full_model_name(cfg) =>
            "Kinetics_c2_I3D_NLN_8x8_R50_BG-KNN_32x32_SultaniBaseline_SultaniLoss_MIL_Baseline"
    """
    return get_dataset_features_name(cfg) + "_" + cfg.MODEL.MODEL_NAME + \
        "_" + cfg.MODEL.LOSS_FUNC + "_" + cfg.TRAIN.TYPE + "_" + cfg.MODEL.SIGN
