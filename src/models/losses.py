"""Define the model losses"""

import torch

from src.models.build import MODEL_REGISTRY


def get_loss_class(cfg, pseudo_label_loss=False):
    """
    Returns the loss class based on cfg.MODEL.LOSS_FUNC
    Args:
        cfg (cfgNode): Model Configurations
        pseudo_label_loss (Bool): If True returns the Pseudo label loss
    Returns:
        loss_class (LossClass): The required loss class
    """
    if pseudo_label_loss:
        loss_class = cfg.MODEL.PSEUDO_LOSS_FUNC
    else:
        loss_class = cfg.MODEL.LOSS_FUNC

    assert loss_class in MODEL_REGISTRY

    return MODEL_REGISTRY.get(loss_class)


@MODEL_REGISTRY.register()
class SultaniLoss():
    """
    Loss exactly as in sultani paper https://arxiv.org/abs/1801.04264v3
    """

    def __init__(self, cfg, preds):
        """
        Construct SultaniLoss
        Args:
            cfg (cfgNode): Model configurations
            preds (List(torch.Tensor)): 2 x Outputs of shape (batch_size x number_of_segments x 1)
                of model with
                    (1) normal input
                    (2) anomaly input
        Note:
            We don't need the L2 regularization on model since optimizer weight decay achieves that
        """
        self.preds_normal = preds[0].squeeze(-1) # shape ((batch_size x number_of_segments)
        self.preds_anomaly = preds[1].squeeze(-1) # shape (batch_size x number_of_segments)

        self.smoothness_lambda = 8e-5
        self.sparisty_lambda = 8e-5

        self.cfg = cfg


    def hinge_ranking_loss(self):
        """Calculates the hinge ranking loss"""
        normal_maxis = self.preds_normal.max(dim=-1).values
        anomaly_maxis = self.preds_anomaly.max(dim=-1).values

        hinge_loss = 1 - anomaly_maxis + normal_maxis
        hinge_loss = torch.max(hinge_loss, torch.zeros_like(hinge_loss))

        return hinge_loss


    def smoothness_loss(self):
        """Calculates the smoothness score"""
        smoothed_scores = self.preds_anomaly[:, 1:] - self.preds_anomaly[:, :-1]
        smoothed_scores_sum_squared = smoothed_scores.pow(2).sum(dim=-1)

        return smoothed_scores_sum_squared


    def sparity_loss(self):
        """Calculates the sparisty loss"""
        return self.preds_anomaly.sum(dim=-1)


    def overall_loss(self):
        """The overall sultani loss"""
        return (
            self.hinge_ranking_loss() + \
            self.smoothness_lambda * self.smoothness_loss() + \
            self.sparisty_lambda * self.sparity_loss()
        ).mean()


    def __call__(self):
        return self.overall_loss()


@MODEL_REGISTRY.register()
class PseudoLabelsLoss():
    """
    Loss calculated via pseduo labels
    """

    def __init__(self, cfg, preds):
        """
        Construct SultaniLoss
        Args:
            cfg (cfgNode): Model configurations
            preds (List(torch.Tensor)): 4 x Outputs of shape (batch_size x number_of_segments x 1)
                of model with:
                    (1) original normal input
                    (2) original anomaly input
                    (3) augmented normal input
                    (4) augmented anomaly input
        Note:
            We don't need the L2 regularization on model since optimizer weight decay achieves that
        """
        self.preds_org_normal = preds[0].squeeze(-1) # shape ((batch_size x number_of_segments)
        self.preds_org_anomaly = preds[1].squeeze(-1) # shape (batch_size x number_of_segments)
        self.preds_aug_normal = preds[2].squeeze(-1) # shape ((batch_size x number_of_segments)
        self.preds_aug_anomaly = preds[3].squeeze(-1) # shape (batch_size x number_of_segments)

        self.cfg = cfg
        self.threshold = cfg.TRAIN.PL_THRESHOLD

    def hinge_ranking_loss(self):
        """Calculates the hinge ranking loss"""
        normal_maxis = self.preds_normal.max(dim=-1).values
        anomaly_maxis = self.preds_anomaly.max(dim=-1).values

        hinge_loss = 1 - anomaly_maxis + normal_maxis
        hinge_loss = torch.max(hinge_loss, torch.zeros_like(hinge_loss))

        return hinge_loss


    def smoothness_loss(self):
        """Calculates the smoothness score"""
        smoothed_scores = self.preds_anomaly[:, 1:] - self.preds_anomaly[:, :-1]
        smoothed_scores_sum_squared = smoothed_scores.pow(2).sum(dim=-1)

        return smoothed_scores_sum_squared


    def sparity_loss(self):
        """Calculates the sparisty loss"""
        return self.preds_anomaly.sum(dim=-1)


    def overall_loss(self):
        """The overall sultani loss"""
        return (
            self.hinge_ranking_loss() + \
            self.smoothness_lambda * self.smoothness_loss() + \
            self.sparisty_lambda * self.sparity_loss()
        ).mean()


    def __call__(self):
        return self.overall_loss()