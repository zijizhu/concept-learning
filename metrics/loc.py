from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as f
from torch import nn
from tqdm import tqdm

from data.cub.cub_dataset import CUBDataset
from data.cub.crop import bbox_to_square_bbox


def in_bbox(point: tuple[float, float], bbox: tuple[float, float, float, float]) -> bool:
    cx, cy, h, w = bbox
    x, y = point
    x_in_bbox = (cx - w / 2) <= x <= (cx + w / 2)
    y_in_bbox = (cy - h / 2) <= y <= (cy + h / 2)
    if x_in_bbox and y_in_bbox:
        return True
    return False


@torch.inference_mode()
def loc_eval(keypoint_annotations: dict,
             model: nn.Module,
             dataset_test: CUBDataset,
             output_dir: str | Path,
             cropped: bool = False,
             bbox_size: int = 90,
             device: torch.device = torch.device("cpu")):
    """dataset_test is expected to pixel_values produced by only pil_to_tensor without any other transforms"""
    transforms = A.Compose([A.Resize(height=232, width=232),
                            A.CenterCrop(height=224, width=224),
                            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])],
                           keypoint_params=A.KeypointParams(format='xy',
                                                            label_fields=['class_labels'],
                                                            remove_invisible=True))

    part_name_list = keypoint_annotations["processed_part_names"]
    image_id_to_keypoints = keypoint_annotations['keypoint_annotations']
    attr_part_map = keypoint_annotations["attribute_part_map"]

    attribute_df = dataset_test.attributes_df

    attr_corrects = np.zeros(len(attribute_df.index))
    attr_total = np.zeros(len(attribute_df.index))

    part_corrects = {p: 0 for p in part_name_list}
    part_total = {p: 0 for p in part_name_list}

    bbox_annotations = dataset_test.bbox_ann
    print(bbox_annotations)

    for sample in tqdm(dataset_test):
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
            bbox = bbox_annotations[image_id]
            x_min, y_min, x_max, y_max = bbox_to_square_bbox(bbox)
            crop_transform = A.Crop(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)
            keypoints = crop_transform.apply_to_keypoints([(x, y, 0.0, 0.0,) for x, y in keypoints])
            keypoints = [(x, y,) for x, y, _, _ in keypoints]

        transformed = transforms(image=sample["pixel_values"].numpy().transpose(1, 2, 0),
                                 keypoints=keypoints,
                                 class_labels=keypoint_labels)
        image = transformed["image"]
        keypoints = transformed["keypoints"]
        keypoint_labels = transformed["class_labels"]

        # `part_keypoint_map` is a dict of visible part name to ground truth keypoint of the image
        part_keypoint_map = {}  # type: dict[str, tuple[int, int]]
        for part_name, kp in zip(keypoint_labels, keypoints):
            part_keypoint_map[part_name] = kp

        # Get set of ground truth active concepts that has available keypoint labels
        active_attr_mask = sample["attr_scores"].numpy().astype(bool)
        active_attr_ids = attribute_df.loc[active_attr_mask, "attribute_id"]  # type: pd.Series

        # Inference
        image_input = f.to_tensor(image).unsqueeze(0).to(device=device)
        outputs = model(image_input)  # type: dict[str, torch.Tensor]

        # Interpolate the attention map to input size of the model
        attn_maps = outputs["attn_maps"].detach().cpu().squeeze().numpy().transpose(1, 2, 0)  # type: np.ndarray
        attn_maps_interpolated = cv2.resize(attn_maps, (224, 224))

        # Loop through all attribute ids in dataset, no matter whether they are used for training
        for attr_id in attr_part_map:
            attr_part_name = attr_part_map[attr_id]
            # Skip if:
            # 1. Sample does not contain ground truth keypoint label of the part corresponding to attribute
            # 2. It is not an active ground truth attribute of the image
            if (attr_part_name not in part_keypoint_map) or (attr_id not in active_attr_ids.values):
                continue

            attn_map_idx = active_attr_ids[active_attr_ids == attr_id].index[0]
            attr_keypoint_label = part_keypoint_map[attr_part_name]

            # Get the coordinate of max attention of the attribute
            # Check if the ground truth keypoint of the attribute is inside the bounding box around the coordinate
            attr_attn_map = attn_maps_interpolated[:, :, attn_map_idx]
            x, y = np.unravel_index(np.argmax(attr_attn_map), attr_attn_map.shape)
            bbox = (x, y, bbox_size, bbox_size,)

            if in_bbox(attr_keypoint_label, bbox):
                attr_corrects[attn_map_idx] += 1
                part_corrects[attr_part_name] += 1

            attr_total[attn_map_idx] += 1
            part_total[attr_part_name] += 1

    # Filter attributes that correspond to parts that are visible in some image samples
    evaluated_attr_mask = attr_total > 0

    # Compute accuracy
    attr_loc_accuracy = np.mean(attr_corrects[evaluated_attr_mask] / attr_total[evaluated_attr_mask])
    part_loc_accuracy = sum(part_corrects.values()) / sum(part_total.values())

    # Save computed results
    np.savez(output_dir / "attribute_localization_results",
             mean_attr_loc_acc=attr_loc_accuracy,
             attr_corrects=attr_corrects,
             attr_total=attr_total,
             evaluated_attr_mask=evaluated_attr_mask)

    attribute_df.to_csv(Path(output_dir) / "attribute_df.csv")

    pd.DataFrame([part_corrects, part_total],
                 columns=[part_name_list],
                 index=["correct", "total"]).to_csv(Path(output_dir) / "part_localization_results.csv")

    # Print Computed results
    print("Number of correct localization:")
    print(attr_corrects)
    print("Number of appearance as ground truth:")
    print(attr_total)
    print("evaluated_attr_mask:")
    print(evaluated_attr_mask)

    print("attr_loc_accuracy:", attr_loc_accuracy)
    print("part_loc_accuracy:", part_loc_accuracy)
