import pandas as pd
import numpy as np
from pathlib import Path
import torch
import pickle as pkl
from torch import nn
import matplotlib.pyplot as plt
import albumentations as A
from torch.utils.data import DataLoader
import torchvision.transforms.functional as f
import cv2
import logging

import argparse

from data.cub.cub_dataset import CUBDataset
from torch.utils.tensorboard import SummaryWriter


def in_bbox(point: tuple[float, float], bbox: tuple[float, float, float, float]) -> bool:
    cx, cy, h, w = bbox
    x, y = point
    x_in_bbox = (cx - w / 2) <= x <= (cx + w / 2)
    y_in_bbox = (cy - h / 2) <= y <= (cy + h / 2)
    if x_in_bbox and y_in_bbox:
        return True
    return False


logger = logging.getLogger(__name__)


@torch.inference_mode()
def loc_eval(keypoint_annotations: dict,
             model: nn.Module,
             dataset_test: CUBDataset,
             writer: SummaryWriter,
             output_dir: str | Path,
             cropped: bool = False,
             bbox_size: int = 90,
             device: torch.device = torch.device("cpu")):
    # TODO incorporate cropping

    transforms = A.Compose([A.Resize(height=232, width=232),
                            A.CenterCrop(height=224, width=224)],
                           keypoint_params=A.KeypointParams(format='xy',
                                                            label_fields=['class_labels'],
                                                            remove_invisible=True))

    part_name_list = keypoint_annotations["processed_part_names"]
    image_id_to_keypoints = keypoint_annotations['keypoint_annotations']
    attr_part_map = keypoint_annotations["attr_part_map"]

    attribute_df = dataset_test.attributes_df

    attr_corrects = np.zeros(len(attribute_df))
    attr_total = np.zeros(len(attribute_df))

    part_corrects = {p: 0 for p in part_name_list}
    part_total = {p: 0 for p in part_name_list}

    bbox_annotations = dataset_test.bbox_ann

    dataloader = DataLoader(dataset_test, batch_size=1, shuffle=False)

    for sample in dataloader:
        # Get visible part keypoints of the image
        image_id = sample["image_ids"].item()
        keypoints = []
        keypoint_labels = []
        for i, kp in enumerate(image_id_to_keypoints[image_id]):
            if kp is not None:
                x, y = np.squeeze(kp)
                keypoints.append((x, y,))
                keypoint_labels.append(part_name_list[i])
        assert len(keypoints) == len(keypoint_labels)

        # Transform input image and keypoints

        # If the dataset uses cropped image, it loads cropped image directly
        # And we need to transform keypoints based on bbox
        if cropped:
            x, y, w, h = bbox_annotations[image_id]
            crop_transform = A.Crop(x_min=x, y_min=y, x_max=x + w, y_max=y + h)
            keypoints = crop_transform.apply_to_keypoints([(x, y, 0.0, 0.0,) for x, y in keypoints])
            keypoints = [(x, y,) for x, y, _, _ in keypoints]
        transformed = transforms(image=sample["pixel_values"].numpy().transpose(1, 2, 0),
                                 keypoints=keypoints,
                                 class_labels=keypoint_labels)
        image = transformed["image"]
        keypoints = transformed["keypoints"]
        keypoint_labels = transformed["keypoint_labels"]

        # `part_keypoint_map` is a dict of visible part name to ground truth keypoint of the image
        part_keypoint_map = {}  # type: dict[str, tuple[int, int]]
        for part_name, kp in zip(keypoint_labels, keypoints):
            part_keypoint_map[part_name] = kp

        # Get set of ground truth active concepts that has available keypoint labels
        active_attr_mask = sample["attr_scores"].numpy().astype(bool)
        active_attr_ids = attribute_df.loc[active_attr_mask, "attribute_id"]
        active_attr_part_names = active_attr_ids.map(attr_part_map.get)

        # Inference
        outputs = model(f.to_tensor(image).to(device=device))  # type: dict[str, torch.Tensor]

        # Interpolate the attention map to input size of the model
        attn_maps = outputs["attn_maps"].detach().squeeze().numpy().transpose(1, 2, 0)  # type: np.ndarray
        attn_maps_interpolated = cv2.resize(attn_maps, (224, 224))

        for idx, attr_id in enumerate(active_attr_ids):
            attr_part_name = attr_part_map[attr_id]
            attr_keypoint_label = part_keypoint_map[attr_part_name]

            # Get the coordinate of max attention of the attribute
            # Check if the ground truth keypoint of the attribute is inside the bounding box around the coordinate
            attr_attn_map = attn_maps_interpolated[:, :, attr_id]
            x, y = np.unravel_index(np.argmax(attr_attn_map), attr_attn_map.shape)
            bbox = (x, y, bbox_size, bbox_size,)

            if in_bbox(attr_keypoint_label, bbox):
                attr_corrects[idx] += 1
                part_corrects[attr_part_name] += 1

            attr_total[idx] += 1
            part_total[attr_part_name] += 1

    # Calculate localization performance and save result
    attr_loc_accuracy = np.mean(attr_corrects / attr_total)
    np.savez(output_dir / "attribute_loc_accuracy.npz",
             attr_loc_accuracy=attr_loc_accuracy,
             attr_corrects=attr_corrects,
             attr_total=attr_total)
    part_loc_accuracy = sum(part_corrects.values()) / sum(part_total.values())
    np.savez(output_dir / "part_loc_accuracy.npz",
             part_loc_accuracy=part_loc_accuracy,
             **{f"{k}_corrects": v for k, v in part_corrects.items()},
             **{f"{k}_total": v for k, v in part_total.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Script to evaluate localization")

    dataset_test = CUBDataset(Path("datasets") / "CUB", split="train_val",
                              use_attrs="data/cub/attributes.txt",
                              use_attr_mask="data/cub/cbm_attribute_mask.txt",
                              use_splits="data/cub/split_image_ids.npz", transforms=None)

    with open("data/cub/attr_part_keypoint_anns.pkl", 'rb') as fp:
        keypoint_annotations = pkl.load(fp)
