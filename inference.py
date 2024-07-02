#!/usr/bin/env python3

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from lightning import seed_everything
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import MulticlassAccuracy
from torchvision.models import ResNet101_Weights, resnet101
from tqdm.notebook import tqdm

from data.cub.cub_dataset import CUBDataset
from data.cub.transforms import get_transforms_dev
from models.loc import SingleBranchModel


def visualize_attn_map(attn_map: np.ndarray, image: Image.Image):
    attn_map_max = np.max(attn_map)
    attn_map_min = np.min(attn_map)
    scaled_map = (attn_map - attn_map_min) / (attn_map_max - attn_map_min)
    heatmap = cv2.applyColorMap(np.uint8(255 * scaled_map), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = heatmap.astype(np.float32) / 255

    return 0.5 * heatmap + np.float32(np.array(image) / 255) * 0.5


prog = """Inference script to extract a visualization of attention map for each attribute"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog)
    parser.add_argument("--test_accuracy", action="store_true")
    parser.add_argument("--out_dir", type=str, default="visualizations")

    seed_everything(42)

    args = parser.parse_args()

    train_transforms, test_transforms = get_transforms_dev(cropped=False)
    dataset_test = CUBDataset(Path("datasets") / "CUB",
                            split="test",
                            use_attrs="continuous",
                            use_attr_mask=None,
                            use_augmentation="augment",
                            use_splits="data/cub/split_image_ids.npz",
                            transforms=test_transforms)

    dataloader_test = DataLoader(dataset=dataset_test, batch_size=1, shuffle=True)

    backbone = resnet101(weights=ResNet101_Weights.DEFAULT)

    net = SingleBranchModel(backbone, class_embeddings=torch.zeros(200, 312))
    state_dict = torch.load("logs/CUB_runs/loc_train_CUB_base/dev_resnet101.pth", map_location="cpu")
    net.load_state_dict(state_dict)
    net.eval()

    if args.test_accuracy:
        mca = MulticlassAccuracy(200, average="micro")
        with torch.no_grad():
            for batch_inputs in tqdm(dataloader_test):
                out = net(batch_inputs['pixel_values'])
                mca(out["class_preds"], batch_inputs["class_ids"])
            
        print(f"Test Acc: {mca.compute().item():.4f}")

    attribute_activations = []
    attn_maps = []
    image_ids = []
    with torch.no_grad():
        for batch_inputs in tqdm(dataloader_test):
            out = net(batch_inputs['pixel_values'])

            attribute_activations.append(out["attr_preds"].numpy())
            attn_maps.append(out["attn_maps"])
            image_ids.append(batch_inputs["image_ids"])
    
    attribute_activations = np.concatenate(attribute_activations)
    attn_maps = np.concatenate(attn_maps)
    image_ids = np.concatenate(image_ids, axis=0)

    resize_transforms = T.Compose([
        T.Resize(232),
        T.CenterCrop(224)
    ])

    attribute_df = dataset_test.attribute_df

    writer = SummaryWriter(log_dir=args.out_dir)

    for attr_id in tqdm(range(312)):
        max_activation_idx = np.argmax(attribute_activations[:, attr_id])

        attn_map = attn_maps[max_activation_idx, attr_id, :, :]
        attn_map_resized = cv2.resize(attn_map, (224, 224,))

        im_id = image_ids[max_activation_idx]
        filename = dataset_test.main_df.loc[im_id, "filename"]
        path_to_image = Path("datasets") / "CUB" / "CUB_200_2011" / "images" / filename
        im = Image.open(path_to_image).convert("RGB")
        img_transformed = resize_transforms(im)

        vis = visualize_attn_map(attn_map_resized, img_transformed)
        attribute_name = attribute_df.loc[attr_id, "attribute"]
        writer.add_image(f"{attribute_name} | {filename}", vis, dataformats='HWC')
