"""Model testing engine"""

import torch
from tqdm import tqdm

from src.models import metrics
from src.utils import funcutils

@torch.no_grad()
@funcutils.debug(apply=False, sign=True, ret=True, sign_beautify=True, ret_beautify=True)
def test(cfg, model, dataloader, print_stats=False):
    """
    Test model on data loader
    Args:
        cfg (cfgNode): Model configurations
        model (torch.nn.model): Video model
        dataloader (DatasetLoader): testing dataset loader
        print_stats (Bool): Whether to print stats or not
    Returns:
        auc (float): Area under the ROC curve
        fpr (numpy.ndarray): False postive rate
        tpr (numpy.ndarray): True positive rate
        thresholds (numpy.ndarray): Thresholds for ROC curve
        acc (float): The overall accuracy
        acc_normal (float): The normal (class = 0) prediction accuracy
        acc_anomaly (float): The anomaly (class = 1) prediction accuracy
    """
    model.eval()

    preds_list = []
    gt_list = []

    if print_stats:
        progress_bar = tqdm(
            total=len(dataloader),
            desc="Testing Progress",
            unit="batch",
            colour="red"
        )

    for data_batch in dataloader:
        features_batch = data_batch["features_batched"]

        if cfg.NUM_GPUS > 0:
            features_batch = features_batch.cuda()

        preds = model(features_batch).squeeze(-1)
        ground_truth = data_batch["annotations"]

        preds_list.append(preds.detach().cpu())
        gt_list.append(ground_truth.detach().cpu())

        if print_stats:
            progress_bar.update(n=1)

    if print_stats:
        progress_bar.close()

    y_true = torch.flatten(torch.cat(gt_list))
    y_probs = torch.flatten(torch.cat(preds_list))

    auc, fpr, tpr, thresholds = metrics.roc_auc(y_true, y_probs)
    acc, acc_normal, acc_anomaly = metrics.accuracy(y_true, y_probs)

    return auc, fpr, tpr, thresholds, acc, acc_normal, acc_anomaly
