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

        self.smoothness_lambda = cfg.LOSS.SL_SMOOTHNESS_LAMBDA
        self.sparisty_lambda = cfg.LOSS.SL_SPARISTY_LAMBDA

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


    def get_progress_bar_info(self):
        """
        Returns progress bar information to be updated per batch
        """
        return {}


@MODEL_REGISTRY.register()
class PseudoLabelsLoss():
    """
    Loss calculated via pseudo labels
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

        self.examples_length = None
        self.cfg = cfg
        self.threshold = cfg.TRAIN.PL_THRESHOLD


    def calculate_positive_anomalies(self):
        """
        Gets the positive aug anomalies whos org score above the threshold
        Returns:
            pos_anomalies (Tensors): Positive augmented anomalies whos
                original values exceed a threshold
            segments_len (int): The length of pos_anomalies
        """
        pos_anomalies = self.preds_aug_anomaly[self.preds_org_anomaly >= self.threshold]
        segments_len = len(pos_anomalies)

        self.examples_length = segments_len

        return pos_anomalies, segments_len


    def calculate_topk_normals(self, topk):
        """
        Calculates the top k (k = segments_len) negative normals
        The idea is to collect the hardest examples with greatest false alarm
        Args:
            topk (int): Number of top segments to be selected
            corrected_top_k (int): The corrected length of pos_anomalies
        """
        assert self.cfg.TRAIN.PL_NORMAL_LABEL_SRC in ["ORG", "AUG"]
        assert self.cfg.TRAIN.PL_NORMAL_LABEL_DST in ["ORG", "AUG"]

        ## To avoid NAN loss
        if topk <= 10:
            topk = 10

        if self.cfg.TRAIN.PL_NORMAL_LABEL_SRC == "ORG":
            indices = torch.topk(self.preds_org_normal.view(-1), topk, largest=False).indices
        elif self.cfg.TRAIN.PL_NORMAL_LABEL_SRC == "AUG":
            indices = torch.topk(self.preds_aug_normal.view(-1), topk, largest=False).indices

        ## Could use preds_org_normal ???
        if self.cfg.TRAIN.PL_NORMAL_LABEL_DST == "ORG":
            return self.preds_org_normal.view(-1)[indices], topk
        elif self.cfg.TRAIN.PL_NORMAL_LABEL_DST == "AUG":
            return self.preds_aug_anomaly.view(-1)[indices], topk


    def overall_loss(self):
        """The overall pseudo labels loss"""
        preds_anomalies, segments_len = self.calculate_positive_anomalies()
        preds_normal, corrected_top_k = self.calculate_topk_normals(segments_len)
        preds = torch.cat([preds_anomalies, preds_normal])

        targets_anomalies = torch.ones(segments_len, dtype=torch.float32)
        targets_normal = torch.zeros(corrected_top_k, dtype=torch.float32)
        targets = torch.cat([targets_anomalies, targets_normal])

        binary_cross_entropy_loss = torch.nn.BCELoss()
        return binary_cross_entropy_loss(preds, targets)

    def __call__(self):
        return self.overall_loss()


    def get_progress_bar_info(self):
        """
        Returns progress bar information to be updated per batch
        """
        all_segments_number = 1
        for dim in self.preds_org_normal.size():
            all_segments_number *= dim

        return {
            "PLs/Anomaly": "{0:.1f}%".format(100.0 * self.examples_length / all_segments_number)
        }
