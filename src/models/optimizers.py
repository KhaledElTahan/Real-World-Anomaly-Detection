"""Define the model optimizers"""

from torch.optim import Adagrad, Adam, Adadelta, SGD

def get_optimizer(cfg, model):
    """
    Utility to get optimizer based on the model configurations
    Args:
        cfg (cfgNode): Model configurations
        model (torch.nn.mode): The trainable model to be optimized
    Returns:
        optimizer (torch.optim.optimizer)
    """

    assert cfg.OPTIMIZER.NAME in ["Adagrad", "Adam", "Adadelta", "SDG"]

    if cfg.OPTIMIZER.NAME == "Adagrad":
        optimizer = Adagrad(
            model.parameters(),
            lr=cfg.OPTIMIZER.BASE_LR,
            eps=cfg.OPTIMIZER.EPS,
            weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY
        )
    elif cfg.OPTIMIZER.NAME == "Adam":
        optimizer = Adam(
            model.parameters(),
            lr=cfg.OPTIMIZER.BASE_LR,
            eps=cfg.OPTIMIZER.EPS,
            weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY
        )
    elif cfg.OPTIMIZER.NAME == "Adadelta":
        optimizer = Adadelta(
            model.parameters(),
            lr=cfg.OPTIMIZER.BASE_LR,
            eps=cfg.OPTIMIZER.EPS,
            weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY
        )
    elif cfg.OPTIMIZER.NAME == "SGD":
        optimizer = SGD(
            model.parameters(),
            lr=cfg.OPTIMIZER.BASE_LR,
            weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY
        )

    return optimizer
