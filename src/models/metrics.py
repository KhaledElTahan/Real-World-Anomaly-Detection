"""Define the evaluation metrics"""

from sklearn import metrics
import torch


def roc_auc(y_true, y_probs, positive_label=1):
    """
    Calculates the Area under the curve of ROC
    Args
        y_true (torch.Tensor or ndarray or list): Ground truth of the example
        y_probs (torch.Tensor or ndarray or list): Predicted probabilities from the model
        positive_label (int): The label of the positive class
    Returns:
        auc (float): Area under the ROC curve
        fpr (numpy.ndarray): False postive rate
        tpr (numpy.ndarray): True positive rate
        thresholds (numpy.ndarray): Thresholds for ROC curve
    """
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_probs, pos_label=positive_label)
    auc = metrics.auc(fpr, tpr)

    return auc, fpr, tpr, thresholds


def accuracy(y_true, y_probs, class_threshold=0.5):
    """
    Calculates the prediction accuracy, normal prediction accuracy, anomaly prediction threshold
    Args
        y_true (torch.Tensor or ndarray or list): Ground truth of the example
        y_probs (torch.Tensor or ndarray or list): Predicted probabilities from the model
        class_threshold (float): Class prediction threshold
    Returns:
        acc (float): The overall accuracy
        acc_normal (float): The normal (class = 0) prediction accuracy
        acc_anomaly (float): The anomaly (class = 1) prediction accuracy
    """
    y_preds = y_probs.clone()

    y_preds[y_preds >= class_threshold] = 1
    y_preds[y_preds < class_threshold] = 0
    y_preds = y_preds.int()

    acc = torch.sum(y_true == y_preds) / torch.numel(y_true)

    normal_indices = (y_true == 0)
    acc_normal = 1.0 - torch.sum(y_preds[normal_indices]) / torch.sum(normal_indices)

    anomaly_indices = (y_true == 1)
    acc_anomaly = torch.sum(y_preds[anomaly_indices]) / torch.sum(anomaly_indices)

    return acc.item(), acc_normal.item(), acc_anomaly.item()
