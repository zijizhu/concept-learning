import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from lightning import seed_everything
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard.writer import SummaryWriter
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm

from data.cub.cub_dataset import CUBDataset
from data.cub.transforms import get_transforms_dev
from metrics.loc import loc_eval
from models.dev import DevModel


def main():
    parser = argparse.ArgumentParser(description="Evaluation Script")
    parser.add_argument("-e", "--experiment_dir", type=str, required=True)

    metric_items = ["accuracy", "full_int_acc", "int_acc_curve", "part_iou"]
    parser.add_argument("-m", "--metrics", nargs="+", choices=metric_items,
                        default=metric_items)

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

    #################
    # Setup logging #
    #################

    summary_writer = SummaryWriter(log_dir=str(log_dir))
    summary_writer.add_text("Model", cfg.MODEL.NAME)
    summary_writer.add_text("Dataset", cfg.DATASET.NAME)
    summary_writer.add_text("Seed", str(cfg.SEED))
    summary_writer.add_text("Device", str(device))

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "eval.log")),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    logger = logging.getLogger(__name__)

    #################################
    # Setup datasets and transforms #
    #################################

    augmentation = cfg.DATASET.get("AUGMENTATION", None)
    _, test_transforms = get_transforms_dev(cropped=bool(augmentation))

    num_attrs = cfg.DATASET.get("NUM_ATTRS", 112)

    num_classes = 200
    # Loads cropped test images if model trained with aug
    dataset_test = CUBDataset(Path(cfg.DATASET.ROOT_DIR) / "CUB", split="test",
                                use_attrs=cfg.DATASET.USE_ATTRS, use_attr_mask=cfg.DATASET.USE_ATTR_MASK,
                                use_splits=cfg.DATASET.USE_SPLITS, use_augmentation=augmentation,
                                transforms=test_transforms)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=cfg.OPTIM.BATCH_SIZE,
                                    shuffle=True, num_workers=8)

    ###############
    # Load models #
    ###############

    from torchvision.models import ResNet101_Weights, resnet101
    backbone = resnet101(weights=ResNet101_Weights.DEFAULT)

    net = DevModel(backbone, num_attrs=num_attrs, num_classes=num_classes,
                   use_attention=cfg.MODEL.USE_ATTENTION)
    state_dict = torch.load(log_dir / f"{cfg.MODEL.NAME}.pth", map_location=device)
    net.load_state_dict(state_dict)

    ###############
    # Evaluations #
    ###############

    net.to(device)
    net.eval()

    # Test Accuracy
    
    mca = MulticlassAccuracy(num_classes=num_classes).to(device)

    attribute_activations = []
    for batch_inputs in tqdm(dataloader_test):
        batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
        outputs = net(batch_inputs['pixel_values'])

        mca(outputs["class_preds"], batch_inputs["class_ids"])

    summary_writer.flush()
    summary_writer.close()
    logger.info("DONE!")


if __name__ == "__main__":
    main()
