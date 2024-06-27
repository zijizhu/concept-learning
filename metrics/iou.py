import os
from pathlib import Path
from collections import defaultdict
import torch
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
import cv2
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from tqdm.notebook import tqdm
from torchvision import ops

from cub_datast import CUBDataset

from model.model_proto import resnet_proto_IoU

from lightning import seed_everything

seed_everything(4896)


part_to_group = {'head': ['has_bill_color',
  'has_eye_color',
  'has_crown_color',
  'has_forehead_color',
  'has_bill_length',
  'has_bill_shape',
  'has_nape_color',
  'has_head_pattern'],
 'breast': ['has_throat_color', 'has_breast_pattern', 'has_breast_color'],
 'belly': ['has_underparts_color', 'has_belly_color', 'has_belly_pattern'],
 'back': ['has_upperparts_color', 'has_back_pattern', 'has_back_color'],
 'wing': ['has_wing_pattern', 'has_wing_color', 'has_wing_shape'],
 'tail': ['has_tail_shape',
  'has_tail_pattern',
  'has_upper_tail_color',
  'has_under_tail_color'],
 'leg': ['has_leg_color'],
 'others': ['has_primary_color', 'has_shape', 'has_size']}

part_remap = {
    'head': [1, 4, 5, 6, 9, 10],
    'breast': [3, 14],
    'belly': [2],
    'back': [0],
    'wing': [8, 12],
    'tail': [13],
    'leg': [7, 11],
    "others": []
}


dataset_test = CUBDataset(dataset_dir=Path("datasets") / "CUB_200_2011",
                          split="test",
                          use_augmentation=None,
                          use_attr_mask=None,
                          use_attrs="continuous",
                          use_part_group="coarse",
                          use_splits=Path("split_image_ids.npz"),
                          transforms=None)
dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)

keypoints_df = pd.read_csv("datasets/CUB_200_2011/CUB_200_2011/parts/part_locs.txt", 
                                   sep=" ", header=None, names=["image_id", "part_id", "x", "y", "visibility"])
keypoints_df["image_id"] -= 1
keypoints_df["part_id"] -= 1

attribute_df = dataloader_test.dataset.attribute_df  # type: pd.DataFrame


dataloader = dataloader_test
cropped = False,
output_dir = Path("outputs"),
input_size = 224,
device = torch.device("cpu")
visualize = False

class Options:
    def __init__(self) -> None:
        self.resnet_path = './pretrained_models/resnet101_c.pth.tar'
        self.dataset = 'CUB'
        self.avg_pool = False
opt = Options()
model = resnet_proto_IoU(opt)
model.load_state_dict(torch.load('out/CUB_GZSL_id_0.pth', map_location='cpu'))
model.eval()

def net(image, attribute):
    output, pre_attri, attention, _ = model(image, attribute)
    return {"attn_maps": attention["layer4"], "attr_preds": pre_attri["layer4"]}

def get_iou(bbox1, bbox2):
    bbox1 = torch.tensor([bbox1])
    bbox2 = torch.tensor([bbox2])
    return ops.box_iou(bbox1, bbox2).item()

def kp_to_bbox(keypoint, width, height, bounds):
    x, y = keypoint
    xmin, ymin = x - width / 2, y - height / 2
    xmax, ymax = x + width / 2, y + height / 2
    if bounds:
        xmin_bound, ymin_bound, xmax_bound, ymax_bound = bounds
        if xmin < xmin_bound:
            xmin = xmin_bound
            xmax = xmin + width
        if ymin < ymin_bound:
            ymin = ymin_bound
            ymax = ymin + height
        if xmax > xmax_bound:
            xmax = xmax_bound
            xmin = xmax - width
        if ymax > ymax_bound:
            ymax = ymax_bound
            ymin = ymax - width
    return xmin, ymin, xmax, ymax


from scipy import io

apn_mat = io.loadmat("data/CUB/APN.mat")
mat = io.loadmat("att_splits.mat")

fnames = []
for i in range(apn_mat["image_files"].shape[0]):
    entry = Path(apn_mat["image_files"][i][0][0])
    fnames.append("/".join(entry.parts[-2:]))

true_filenames = dataset_test.main_df["filename"].tolist()
unseen_fnames =[fnames[sample_id] for sample_id in np.squeeze(mat["test_unseen_loc"] - 1)]
test_image_ids = [true_filenames.index(fn) for fn in unseen_fnames]
dataset_test.image_ids["test"] = test_image_ids


import matplotlib.patches as patches

NUM_ORIGINAL_PARTS = 15
visualize = False
resize_part_bbox = True

with open("datasets/CUB_200_2011/CUB_200_2011/parts/parts.txt", 'r') as fp:
    parts = fp.read().splitlines()
original_part_list = [p.split(" ", 1)[1] for p in parts]

assert dataloader.batch_size == 1
# If model takes in cropped images, process keypoints
part_bbox_scale = 0.33

tv_transforms = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transforms = A.Compose([
    A.SmallestMaxSize(256),
    A.CenterCrop(height=224, width=224)
], keypoint_params=A.KeypointParams(format='xy', label_fields=['class_labels'], remove_invisible=False),
bbox_params=A.BboxParams(format="coco", clip=True))
normalize = A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

keypoints_df = pd.read_csv("datasets/CUB_200_2011/CUB_200_2011/parts/part_locs.txt",
                           sep=" ", header=None, names=["image_id", "part_id", "x", "y", "visibility"])
keypoints_df["image_id"] -= 1
keypoints_df["part_id"] -= 1

dataloader.dataset.transforms = None
input_shape = (224, 224,)

ious = []
for sample in tqdm(dataloader):
    image_path = sample["filename"][0]
    image_id = sample["image_ids"].item()
    raw_image = Image.open(image_path).convert("RGB")

    # Process keypoint annotations if the model takes in cropped images
    cropped = False
    if cropped:
        bbox = dataloader.dataset.bbox_ann[image_id]
        bbox = dataset_test.bbox_ann[0]
        x, y, w, h = bbox
        crop_transform = A.Crop(x_min=x, y_min=y, x_max=x + w, y_max=y + h)
        crop_transform.apply_to_keypoints([(x, y, 0.0, 0.0,) for x, y in ...])
        pass

    # Apply size transforms to input image and keypoints
    kp_anns = keypoints_df[keypoints_df["image_id"] == image_id]
    keypoints = kp_anns[["x", "y"]].values
    kp_labels = kp_anns["part_id"].values
    kp_visibility = kp_anns["visibility"].values

    bbox_coords = list(int(val) for val in dataloader.dataset.bbox_ann[image_id])

    transformed = transforms(image=np.array(raw_image), keypoints=keypoints,
                             class_labels=kp_labels, bboxes=[bbox_coords + ["bird"]])

    # image_transformed = normalize(image=transformed["image"])["image"]
    keypoints_transformed = np.array(transformed["keypoints"]).astype(int)
    kp_labels_transformed = transformed["class_labels"]
    bbox_xmin, bbox_ymin, bbox_w, bbox_h, _ = transformed["bboxes"][0]

    part_bbox_w, part_bbox_h = bbox_w * part_bbox_scale, bbox_h * part_bbox_scale
    if resize_part_bbox:
        part_bbox_h = max(int(part_bbox_w / 2), part_bbox_h)
        part_bbox_w = max(int(part_bbox_h / 2), part_bbox_w)

    # Inference
    raw_image = Image.open(image_path).convert("RGB")
    outputs = net(tv_transforms(raw_image).unsqueeze(0).to(device), dataset_test.attribute_vectors_pt.T)
    attn_maps = outputs["attn_maps"].squeeze().detach().numpy().transpose(1, 2, 0)
    attr_preds = outputs["attr_preds"].squeeze().detach().numpy()
    attn_maps_resized = cv2.resize(attn_maps, input_shape)

    # Compute a dict that maps each part to a list of attention map indices
    # where each index corresponds to highest attribute prediction of the group
    part_to_map_indices = {}
    for part, group_names in part_to_group.items():
        selected_map_idx_list = []
        for group in group_names:
            group_mask = (attribute_df["attribute_group"] == group).values
            used_mask = attribute_df["model_attribute_index"].notna().values
            mask = group_mask & used_mask

            if np.all(~mask):  # No attribute from this group is selected for training
                continue

            model_attr_indices = attribute_df.loc[mask, "model_attribute_index"].values

            max_pred_idx = np.argmax(attr_preds[model_attr_indices])
            selected_map_idx = model_attr_indices[max_pred_idx]

            selected_map_idx_list.append(selected_map_idx)
        part_to_map_indices[part] = selected_map_idx_list

    part_to_IoUs = defaultdict(list)
    for part, map_indices in part_to_map_indices.items():
        # Compute a mask to select all visible keypoints related to the part
        mask = np.isin(np.arange(NUM_ORIGINAL_PARTS), part_remap[part]) & kp_visibility == 1
        if np.all(~mask):  # None of the keypoints for this part is visible in the image
            continue
        
        # print(part)

        candidate_gt_keypoints = keypoints_transformed[mask]
        for map_idx in map_indices:
            attn_map = attn_maps_resized[:, :, map_idx]
            # Find the part keypoint that is the closest to center of attention map
            attn_cy, attn_cx = np.unravel_index(np.argmax(attn_map), shape=input_shape)

            # print("attention center point:")
            # print(attn_cx, attn_cy)
            dists = np.array([((x - attn_cx) ** 2 + (y - attn_cy) ** 2) for x, y in candidate_gt_keypoints])
            # print("chosen gt idx out of 15:", np.argmin(dists))
            gt_cx, gt_cy = candidate_gt_keypoints[np.argmin(dists)]

            # print("gt center point:")
            # print(gt_cx, gt_cy)

            attn_center_bbox = kp_to_bbox((attn_cx, attn_cy,),
                                          width=part_bbox_w,
                                          height=part_bbox_h,
                                          bounds=(bbox_xmin, bbox_ymin, bbox_xmin + bbox_w, bbox_ymin + bbox_h,))
            attn_xmin, attn_ymin, attn_xmax, attn_ymax = attn_center_bbox
            
            gt_bbox = kp_to_bbox((gt_cx, gt_cy,),
                                 width=part_bbox_w,
                                 height=part_bbox_h,
                                 bounds=(bbox_xmin, bbox_ymin, bbox_xmin + bbox_w, bbox_ymin + bbox_h,))
            gt_xmin, gt_ymin, gt_xmax, gt_ymax = gt_bbox
            
            # print("attn_center_bbox")
            # print(attn_center_bbox)
            # print("gt_bbox:")
            # print(gt_bbox)

            visualize = False
            if visualize:
                plt.imshow(transformed["image"])
                plt.plot(gt_cx, gt_cy, "ro")
                plt.plot(attn_cx, attn_cy, "bo")
                rect1 = patches.Rectangle((gt_xmin, gt_ymin), gt_xmax - gt_xmin, gt_ymax - gt_ymin,
                                        linewidth=1, edgecolor='r', facecolor='none')
                rect2 = patches.Rectangle((attn_xmin, attn_ymin), attn_xmax - attn_xmin, attn_ymax - attn_ymin,
                                        linewidth=1, edgecolor='b', facecolor='none')
                ax = plt.gca()
                ax.add_patch(rect1)
                ax.add_patch(rect2)
            
            iou = get_iou(attn_center_bbox, gt_bbox)
            part_to_IoUs[part].append(iou)
            # print(iou)
    
    # print(part_to_IoUs)
    ious.append(part_to_IoUs)