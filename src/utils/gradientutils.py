"""Utilities related to model gradient"""

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

def plot_grad_flow(cfg, model):
    """
    Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    Usage: Plug this function in Trainer class after loss.backwards()
        to visualize the gradient flow
    Args:
        cfg (cfgNode): Model Configurations
        model (Torch.nn.Module): The model of interest
    """
    ave_grads = []
    max_grads= []
    layers = []

    if cfg.NUM_GPUS > 1:
        model = model.module

    named_parameters = model.named_parameters()

    top_ylim = 0.52
    for name, param in named_parameters:
        if(param.requires_grad) and ("bias" not in name):
            layers.append(name)
            ave_grads.append(param.grad.abs().mean())
            max_grads.append(param.grad.abs().max())

            if param.grad.abs().max() > top_ylim:
                top_ylim = param.grad.abs().max()

    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=top_ylim) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)],
                ['max-gradient', 'mean-gradient', 'zero-gradient']
            )

    plt.show()
