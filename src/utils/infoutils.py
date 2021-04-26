"""Information Utils"""

def get_dataset_features_name(cfg):
    """
    Creates dataset features name
    Args:
        cfg: The video model configuration file
    Returns:
        features_name (String): represents the dataset features name
    Example:
        get_dataset_features_name(cfg) => "Kinetics_c2_I3D_NLN_8x8_R50_BG-KNN_32x32"
    """
    return cfg.BACKBONE.NAME + "_" + cfg.TRANSFORM.CODE + "_" + \
        str(cfg.EXTRACT.FRAMES_BATCH_SIZE) + "x" + str(cfg.EXTRACT.NUMBER_OUTPUT_SEGMENTS)
