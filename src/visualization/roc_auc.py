"""Utility for ROC-AUC Visualization"""

import matplotlib.pyplot as plt

from src.utils import infoutils

def plot_signle_roc_auc(cfg, auc, fpr, tpr):
    """
    Plots signel ROC Curve
    Args:
        cfg (cfgNode): Model configurations
        auc (float): Area under the ROC curve
        fpr (list): False positive rates
        tpr (list): True positive rates
    """
    plt.figure()

    plt.plot(fpr, tpr, color='darkorange',
        lw=cfg.VISUALIZE.PLIT_LINEWIDTH, label='ROC curve (area = %0.6f)' % auc)

    plt.plot([0, 1], [0, 1], color='navy',
        lw=cfg.VISUALIZE.PLIT_LINEWIDTH, label='Random Classifier ROC (area = 0.5)', linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.01])

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.title('Receiver Cperating Characteristic (ROC)' + '\n' + \
        infoutils.get_dataset_features_name(cfg) + '\n' + infoutils.get_full_model_without_features(cfg))

    plt.legend(loc="lower right")

    plt.show()
