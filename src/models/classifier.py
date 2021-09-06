"""Define the video classifier"""

import torch

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
        self.relu1 = torch.nn.ReLU()
        self.dropout1 = torch.nn.Dropout(self.cfg.MODEL.DROPOUT_RATE)

        self.fc2 = torch.nn.Linear(512, 32)
        self.relu2 = torch.nn.ReLU()
        self.dropout2 = torch.nn.Dropout(self.cfg.MODEL.DROPOUT_RATE)

        self.fc3 = torch.nn.Linear(32, 1)
        self.sig = torch.nn.Sigmoid()

        self._initialize_weights()


    def _initialize_weights(self):
        """Initialize the model weights"""
        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)
        torch.nn.init.xavier_normal_(self.fc3.weight)


    def forward(self, inputs):
        """
        Forward inputs to model and produce output
        Args:
            inputs (torch.Tensor): batch of video features
        Returns:
            output (torch.Tensor): probability of segment being anomaleous
        """
        inputs = self.fc1(inputs)
        inputs = self.relu1(inputs)
        inputs = self.dropout1(inputs)

        inputs = self.fc2(inputs)
        inputs = self.relu2(inputs)
        inputs = self.dropout2(inputs)

        inputs = self.fc3(inputs)
        output = self.sig(inputs)

        return output
