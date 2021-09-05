"""Define the video classifier"""

import torch
from torch.functional import F

from src.models.build import MODEL_REGISTRY

@MODEL_REGISTRY.register()
class SultaniBaseline(torch.nn.Module):
    """
    Baseline built exactly as in sultani paper https://arxiv.org/abs/1801.04264v3
    """
    def __init__(self, cfg):
        super(SultaniBaseline, self).__init__()

        self.cfg = cfg

        self.fc1 = torch.nn.Linear(self.cfg.BACKBONE.FEATURES_LENGTH, 512)
        self.fc2 = torch.nn.Linear(512, 32)
        self.fc3 = torch.nn.Linear(32, 1)

        # Define proportion or neurons to dropout
        self.dropout = torch.nn.Dropout(self.cfg.MODEL.DROPOUT_RATE)


    def forward(self, inputs):
        """
        Forward inputs to model and produce output
        Args:
            inputs (torch.Tensor): batch of video features
        Returns:
            output (torch.Tensor): probability of segment being anomaleous
        """
        inputs = self.fc1(inputs)
        inputs = F.relu(inputs)
        inputs = self.dropout(inputs)

        inputs = self.fc2(inputs)
        inputs = F.relu(inputs)
        inputs = self.dropout(inputs)

        inputs = self.fc3(inputs)
        output = F.sigmoid(inputs)

        return output
