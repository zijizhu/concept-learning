import argparse
import logging
import pickle as pkl
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as f
from lightning import seed_everything
from omegaconf import OmegaConf
from torch import nn
from tqdm import tqdm

from data.cub.crop import bbox_to_square_bbox
from data.cub.cub_dataset import CUBDataset
from data.cub.transforms import get_transforms_dev
from models.dev import DevModel

from metrics.loc import loc_eval


def in_bbox(point: tuple[float, float], bbox: tuple[float, float, float, float]) -> bool:
    cx, cy, h, w = bbox
    x, y = point
    x_in_bbox = (cx - w / 2) <= x <= (cx + w / 2)
    y_in_bbox = (cy - h / 2) <= y <= (cy + h / 2)
    if x_in_bbox and y_in_bbox:
        return True
    return False


logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Localization Evaluation Script")
    parser.add_argument("-e", "--experiment_dir", type=str, required=True)

    args = parser.parse_args()
    log_dir = Path(args.experiment_dir)
    config_path = log_dir / "hparams.yaml"
    cfg = OmegaConf.load(config_path)

    seed_everything(cfg.SEED)
    rng = np.random.default_rng(cfg.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    experiment_name = log_dir.stem
    print("Experiment Name:", experiment_name)
    print("Device:", device)

    #################################
    # Setup datasets and transforms #
    #################################

    if cfg.DATASET.NAME == "CUB":
        augmentation = cfg.DATASET.get("AUGMENTATION", None)
        train_transforms, test_transforms = get_transforms_dev(cropped=True if augmentation else False)

        num_attrs = cfg.DATASET.get("NUM_ATTRS", 112)

        num_classes = 200
        dataset_test_no_transform = CUBDataset(Path(cfg.DATASET.ROOT_DIR) / "CUB", split="test",
                                  use_attrs=cfg.DATASET.USE_ATTRS, use_attr_mask=cfg.DATASET.USE_ATTR_MASK,
                                  use_splits=cfg.DATASET.USE_SPLITS, use_augmentation=augmentation,
                                  transforms=None)  # Loads cropped test images if model trained with aug
    else:
        raise NotImplementedError

    ###############
    # Load models #
    ###############

    if cfg.MODEL.BACKBONE == 'resnet101':
        from torchvision.models import resnet101, ResNet101_Weights

        backbone = resnet101(weights=ResNet101_Weights.DEFAULT)
    elif cfg.MODEL.BACKBONE == 'resnet50':
        from torchvision.models import resnet50, ResNet50_Weights

        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
    else:
        raise NotImplementedError

    net = DevModel(backbone, num_attrs=num_attrs, num_classes=num_classes,
                   use_attention=cfg.MODEL.USE_ATTENTION)
    state_dict = torch.load(log_dir / f"{cfg.MODEL.NAME}.pth", map_location=device)
    net.load_state_dict(state_dict)

    with open("data/cub/keypoint_annotations.pkl", "rb") as fp:
        keypoint_anns = pkl.load(fp)
    print(list(keypoint_anns.keys()))

    ###############
    # Evaluations #
    ###############

    net.to(device)
    net.eval()

    loc_eval(keypoint_anns,
             net,
             dataset_test_no_transform,
             Path("./"),
             True if augmentation else False,
             90, device)
