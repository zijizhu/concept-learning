from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from PIL import Image
import cv2


###############################################
# Functions for visualizing prototype weights #
###############################################

def visualize_prototypes_multiple_epochs(experiment_dir: str | Path):
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


def visualize_prototypes_one_epoch(experiment_dir: str | Path):
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


def visualize_attn_map(attn_map: np.ndarray, image: Image.Image):
    attn_map_max = np.max(attn_map)
    attn_map_min = np.min(attn_map)
    scaled_map = (attn_map - attn_map_min) / (attn_map_max - attn_map_min)
    heatmap = cv2.applyColorMap(np.uint8(255 * scaled_map), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = heatmap.astype(np.float32) / 255

    return 0.5 * heatmap + np.float32(np.array(image) / 255) * 0.5
