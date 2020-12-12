"""Utilies used for features extraction from models"""
import torch.nn.functional as F

def create_features(features, detach=True):
    """
    Creates a copy from the features tensor and normalizes it.
    Should be called from network head.
    Args:
        features (torch.Tensor): Output from models pathways
        detach (Boolean): Detach features from model graph
    """
    # save features
    features = features.clone()

    if detach:
        features = features.detach()

    # flatten the features tensor
    features = features.mean(3).mean(2).reshape(features.shape[0], -1)

    # apply l2 normalization on features
    F.normalize(features)

    return features
