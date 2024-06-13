from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def visualize_multiple_epochs(experiment_dir: str):
    fig, axes = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True, figsize=(12, 12))
    experiment_dir = Path(experiment_dir)
    all_prototype_weights = torch.load(experiment_dir / "prototype_weights.pth")
    all_prototype_weights = all_prototype_weights.squeeze()
    n_epochs, b, dim = all_prototype_weights.shape
    vis_epochs = np.linspace(0, n_epochs - 1, num=9).astype(int)
    for ax, epoch in zip(axes.flatten(), vis_epochs):
        epoch_prototype_weights = all_prototype_weights[epoch, :, :].numpy()
        ax.hist(epoch_prototype_weights.flatten(), bins=100)
        ax.set_title(f"epoch {epoch}")
    fig.tight_layout()


def visualize_one_epoch(experiment_dir: str):
    fig, axes = plt.subplots(nrows=12, ncols=10, sharex=True, sharey=True, figsize=(12, 18))
    experiment_dir = Path(experiment_dir)
    all_prototype_weights = torch.load(experiment_dir / "prototype_weights.pth")
    all_prototype_weights = all_prototype_weights.squeeze()
    epoch = -1
    epoch_weights = all_prototype_weights[epoch, :, :].numpy()
    for i, ax in enumerate(axes.flatten()):
        if i < len(epoch_weights):
            ax.hist(epoch_weights[i, :], bins=100)
            ax.tick_params(rotation=90)
            ax.set_title(f"attribute {i}", fontsize=8)
        else:
            ax.set_visible(False)
    fig.tight_layout()
