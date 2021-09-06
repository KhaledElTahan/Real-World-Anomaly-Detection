"""Define the evaluation metrics"""

from sklearn import metrics

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
