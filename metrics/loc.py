import pandas as pd
import numpy as np
from pathlib import Path
import torch
import pickle as pkl
import matplotlib.pyplot as plt
import albumentations as A
import cv2

from data.cub.cub_dataset import CUBDataset
dataset_test = CUBDataset(Path("datasets") / "CUB", split="train_val",
                          use_attrs="data/cub/attributes.txt", use_attr_mask="data/cub/attribute_names_mask.txt",
                          use_splits="data/cub/split_image_ids.npz", transforms=None)

if __name__ == "__main__":
    with open("data/cub/attr_part_keypoint_anns.pkl", 'rb') as fp:
        keypoint_anns = pkl.load(fp)

    attr_corrects = np.zeros(312)
    attr_total = np.zeros(312)

    # TODO If using a subset of attributes, dataset resets index and

    transforms = A.Compose([A.Resize(height=232, width=232),
                            A.CenterCrop(height=224, width=224)],
                           keypoint_params=A.KeypointParams(format='xy',
                                                            label_fields=['class_labels'],
                                                            remove_invisible=True))
    part_name_list = keypoint_anns["processed_part_names"]
    image_id_to_keypoints = keypoint_anns['keypoint_annotations']
    attr_part_map = keypoint_anns["attr_part_map"]

    for sample_idx, sample in enumerate(dataset_test):
        # Get visible part keypoints of the image
        image_id = sample["image_ids"].item()
        keypoints = []
        keypoint_labels = []
        for i, kp in enumerate(image_id_to_keypoints[image_id]):
            if kp is not None:
                x, y = np.squeeze(kp)
                keypoints.append((x, y,))
                keypoint_labels.append(part_name_list[i])

        # Transform input image and keypoints
        # `part_keypoint_map` is a dict of visible part name to ground truth keypoint of the image
        transformed = transforms(image=sample["pixel_values"].numpy().transpose(1, 2, 0),
                                 keypoints=keypoints,
                                 class_labels=keypoint_labels)
        image = transformed["image"]
        keypoints = transformed["keypoints"]
        keypoint_labels = transformed["keypoint_labels"]

        part_keypoint_map = {}  # type: dict[str, tuple[int, int]] # TODO
        for part_name, kp in zip(keypoint_labels, keypoints):
            part_keypoint_map[part_name] = kp

        # Get set of ground truth active concepts that has available keypoint labels
        active_attr_mask = sample["attr_scores"].numpy().astype(bool)
        active_attr_ids = dataset_test.attributes_df.loc[active_attr_mask, "attribute_id"]
        active_attr_part_names = active_attr_ids.map(attr_part_map.get)

        # Inference
        outputs = {}


        # Interpolate attention maps
        def in_bbox(point: tuple[float, float], bbox: tuple[float, float, float, float]) -> bool:
            cx, cy, h, w = bbox
            x, y = point
            x_in_bbox = (cx - w / 2) <= x <= (cx + w / 2)
            y_in_bbox = (cy - h / 2) <= y <= (cy + h / 2)
            if x_in_bbox and y_in_bbox:
                return True
            return False


        # Interpolate the attention map to input size of the model
        attn_maps = outputs["attn_maps"].squeeze().numpy().transpose(1, 2, 0)  # type: np.ndarray
        attn_maps_interpolated = cv2.resize(attn_maps, (224, 224))

        for idx, attr_id in enumerate(active_attr_ids):
            attr_part = attr_part_map[attr_id]
            attr_keypoint_label = part_keypoint_map[attr_part]

            # Get the coordinate of max attention of the attribute
            # Check if the ground truth keypoint of the attribute is inside the bounding box around the coordinate
            attr_attn_map = attn_maps_interpolated[:, :, attr_id]
            x, y = np.unravel_index(np.argmax(attr_attn_map), attr_attn_map.shape)
            bbox = (x, y, 90, 90,)

            if in_bbox(attr_keypoint_label, bbox):
                attr_corrects[idx] += 1

            attr_total[idx] += 1

    # Calculate localization performance
    loc_accuracy = np.mean(attr_corrects / attr_total)