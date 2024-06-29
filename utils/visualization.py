import cv2
import numpy as np
from PIL import Image


def visualize_attn_map(attn_map: np.ndarray, image: Image.Image):
    """Overlay an attention map on image for visualization."""
    attn_map_max = np.max(attn_map)
    attn_map_min = np.min(attn_map)
    scaled_map = (attn_map - attn_map_min) / (attn_map_max - attn_map_min)
    heatmap = cv2.applyColorMap(np.uint8(255 * scaled_map), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = heatmap.astype(np.float32) / 255

    return 0.5 * heatmap + np.float32(np.array(image) / 255) * 0.5
