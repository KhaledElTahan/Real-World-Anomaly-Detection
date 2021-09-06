"""Model testing engine"""

import torch
from tqdm import tqdm

from src.models import metrics
from src.utils import funcutils

@torch.no_grad()
@funcutils.debug(apply=False, sign=True, ret=True, sign_beautify=True, ret_beautify=True)
def test(model, dataloader, print_stats=False):
    """
    Test model on data loader
    Args:
        model (torch.nn.model): Video model
        dataloader (DatasetLoader): testing dataset loader
        print_stats (Bool): Whether to print stats or not
    Returns:
        auc (float): Area under the ROC curve
        fpr (numpy.ndarray): False postive rate
        tpr (numpy.ndarray): True positive rate
        thresholds (numpy.ndarray): Thresholds for ROC curve
    """
    model.eval()

    preds_list = []
    gt_list = []

    if print_stats:
        progress_bar = tqdm(total=len(dataloader), desc="Model Testing Progress")

    for data_batch in dataloader:
        preds = model(data_batch["features_batched"]).squeeze(-1)
        ground_truth = data_batch["annotations"]

        preds_list.append(preds.detach())
        gt_list.append(ground_truth.detach())

        if print_stats:
            progress_bar.update(n=1)

    if print_stats:
        progress_bar.close()

    auc, fpr, tpr, thresholds = metrics.roc_auc(
        torch.flatten(torch.cat(gt_list)),
        torch.flatten(torch.cat(preds_list))
    )

    if print_stats:
        print("Test completed with AUC ", auc)

    return auc, fpr, tpr, thresholds
