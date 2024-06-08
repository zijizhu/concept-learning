import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

import timm
import numpy as np
import torch
from lightning import seed_everything
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from data.cub.cub_dataset import CUBDataset
from data.cub.transforms import get_transforms_dev
from models.utils import Backbone
from models.cbm import CBM
from models.dev import DevModel


def get_hi_lo_activations(model, dataloader):
    ...


# TODO
def test_interventions(model: nn.Module, dataset_test: CUBDataset, num_groups_to_intervene: list[int],
                       num_corrects_fn: Callable, dataset_size: int,
                       rng: np.random.Generator, logger: logging.Logger, writer: SummaryWriter, device: torch.device):
    """Given a dataset and concept learning model, test its ability of responding to test-time interventions"""
    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=4)

    for num_groups in num_groups_to_intervene:
        sampled_group_ids = rng.choice(
            np.arange(len(dataset_test.group_names)),
            size=(len(dataset_test), num_groups)
        )
        running_corrects = 0
        # Inference loop
        for test_inputs, group_ids_to_intervene in tqdm(zip(dataloader_test, sampled_group_ids), total=len(dataloader_test)):
            intervention_mask = np.isin(dataset_test.attribute_group_indices, group_ids_to_intervene)
            intervention_mask = torch.tensor(intervention_mask.astype(int), device=device)
            test_inputs = {k: v.to(device) for k, v in test_inputs.items()}
            results = model.inference(test_inputs, int_mask=intervention_mask, int_values=test_inputs["attr_scores"])

            running_corrects += num_corrects_fn(results, test_inputs)

        # Compute accuracy

        acc = running_corrects / dataset_size
        writer.add_scalar("Acc/train", acc, num_groups)
        logger.info(f"Test Acc: {acc:.4f}")


@torch.no_grad()
def compute_corrects(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]):
    class_preds, class_ids = outputs["class_scores"], batch["class_ids"]
    return torch.sum(torch.argmax(class_preds.data, dim=-1) == class_ids.data).item()


@torch.no_grad()
def test_accuracy(
    model: nn.Module,
    num_corrects_fn: nn.Module | Callable,
    dataloader: DataLoader,
    dataset_size: int,
    device: torch.device,
    logger: logging.Logger,
):
    running_corrects = 0

    for batch_inputs in tqdm(dataloader):
        batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
        outputs = model(batch_inputs['pixel_values'])

        running_corrects += num_corrects_fn(outputs, batch_inputs)

    epoch_acc = running_corrects / dataset_size
    logger.info(f"Test Acc: {epoch_acc:.4f}")

    return epoch_acc


def main():
    parser = argparse.ArgumentParser(description="Evaluation Script")
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

    #################
    # Setup logging #
    #################

    summary_writer = SummaryWriter(log_dir=str(log_dir), comment="eval")
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
    )
    logger = logging.getLogger(__name__)

    #################################
    # Setup datasets and transforms #
    #################################

    #################################
    # Setup datasets and transforms #
    #################################

    if cfg.DATASET.NAME == "CUB":
        train_transforms, test_transforms = get_transforms_dev()

        num_attrs = cfg.get("DATASET.NUM_ATTRS", 112)
        num_classes = 200
        dataset_train = CUBDataset(
            Path(cfg.DATASET.ROOT_DIR) / "CUB", split="train_val", use_attrs=cfg.DATASET.USE_ATTRS,
            use_attr_mask=cfg.DATASET.USE_ATTR_MASK, use_splits=cfg.DATASET.USE_SPLITS,
            transforms=train_transforms)
        dataset_test = CUBDataset(
            Path(cfg.DATASET.ROOT_DIR) / "CUB", split="test", use_attrs=cfg.DATASET.USE_ATTRS,
            use_attr_mask=cfg.DATASET.USE_ATTR_MASK, use_splits=cfg.DATASET.USE_SPLITS,
            transforms=train_transforms)
        dataloader_train = DataLoader(
            dataset=dataset_train, batch_size=1,
            shuffle=True, num_workers=8)
        dataloader_test = DataLoader(
            dataset=dataset_test, batch_size=1,
            shuffle=True, num_workers=8)
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

    net = DevModel(backbone, num_attrs=num_attrs, num_classes=num_classes, activation=cfg.MODEL.ACTIVATION)
    state_dict = torch.load(log_dir / f"{cfg.MODEL.NAME}.pth", map_location=device)
    net.load_state_dict(state_dict)

    ###############
    # Evaluations #
    ###############

    net.to(device)
    net.eval()

    # Test Accuracy
    logger.info("Start task accuracy evaluation...")
    test_accuracy(net, compute_corrects, dataloader_test, len(dataloader_test), device, logger)

    exit(0)

    # TODO Test Intervention
    logger.info("Start intervention evaluation...")

    num_groups_to_intervene = [0, 4, 8, 12, 16, 20, 24, 28]
    test_interventions(model=net, dataset_test=dataset_test, num_groups_to_intervene=num_groups_to_intervene,
                       num_corrects_fn=compute_corrects, dataset_size=len(dataset_test), logger=logger,
                       writer=summary_writer, device=device, rng=rng)

    # TODO Test Representation

    logger.info("DONE!")


if __name__ == "__main__":
    main()
