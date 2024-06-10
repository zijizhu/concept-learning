import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Callable

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
from models.dev import DevModel


@torch.no_grad()
def compute_corrects(outputs: dict[str, torch.Tensor] | torch.Tensor, batch: dict[str, torch.Tensor]):
    if isinstance(outputs, dict):
        class_preds = outputs["class_scores"]
    else:
        class_preds = outputs
    class_ids = batch["class_ids"]
    return torch.sum(torch.argmax(class_preds.data, dim=-1) == class_ids.data).item()


@torch.inference_mode()
def get_lo_hi_activations(model, dataloader, hi_lo_quantiles: tuple[int, int] = (0.95, 0.05),
                          device: torch.device = torch.device('cpu')):
    hi_quantile, lo_quantile = hi_lo_quantiles
    all_concept_activations = []
    for batch in tqdm(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(batch["pixel_values"])
        all_concept_activations.append(outputs["attr_scores"])
    activations = torch.cat(all_concept_activations, dim=0)
    hi_activations = torch.quantile(activations, hi_quantile, dim=0)  # shape: [num_attrs]
    lo_activations = torch.quantile(activations, lo_quantile, dim=0)  # shape: [num_attrs]

    return torch.stack([lo_activations, hi_activations], dim=-1)  # shape: [num_attrs, 2]


@torch.inference_mode()
def test_interventions(model: nn.Module, dataloader: DataLoader, num_int_groups: list[int],
                       attribute_group_indices: np.array, train_activations: torch.Tensor | None, use_sigmoid: bool,
                       num_corrects_fn: Callable, dataset_size: int, rng: np.random.Generator,
                       logger: logging.Logger, writer: SummaryWriter, device: torch.device):
    """Given a dataset and concept learning model, test its ability of responding to test-time interventions"""
    num_attrs = 112

    for num_groups in num_int_groups:
        sampled_group_ids = rng.choice(np.arange(len(np.unique(attribute_group_indices))),
                                       size=(len(dataloader), num_groups))
        running_corrects = 0
        # Inference loop
        for test_inputs, int_group_ids in tqdm(zip(dataloader, sampled_group_ids), total=len(dataloader)):
            int_mask = np.isin(attribute_group_indices, int_group_ids)
            int_mask = torch.tensor(int_mask.astype(int), device=device)
            test_inputs = {k: v.to(device) for k, v in test_inputs.items()}
            if not use_sigmoid:
                int_value_indices = (torch.arange(num_attrs).to(device),
                                     test_inputs["attr_scores"].squeeze().to(dtype=torch.long))
                int_values = train_activations[int_value_indices]
            else:
                int_values = test_inputs["attr_scores"]
            results = model.inference(test_inputs["pixel_values"], int_mask=int_mask, int_values=int_values)

            running_corrects += num_corrects_fn(results, test_inputs)

        # Compute accuracy
        acc = running_corrects / dataset_size
        writer.add_scalar("Acc/intervention curve", acc, num_groups)
        logger.info(f"Test Acc when {num_groups} attribute groups intervened: {acc:.4f}")


@torch.inference_mode()
def test_interventions_full(model: nn.Module, dataloader: DataLoader, num_corrects_fn: Callable,
                            dataset_size: int, logger: logging.Logger, writer: SummaryWriter, device: torch.device):
    """Given a dataset and concept learning model, test its ability of responding to test-time interventions"""
    running_corrects = 0
    # Inference loop
    for test_inputs in tqdm(dataloader):
        test_inputs = {k: v.to(device) for k, v in test_inputs.items()}
        results = model.c2y(test_inputs["attr_scores"].to(torch.float32))

        running_corrects += num_corrects_fn(results, test_inputs)

    # Compute accuracy
    acc = running_corrects / dataset_size
    writer.add_scalar("Acc/full intervention", acc)
    logger.info(f"Test Acc full intervention: {acc:.4f}")


@torch.no_grad()
def test_accuracy(model: nn.Module,
                  num_corrects_fn: nn.Module | Callable,
                  dataloader: DataLoader,
                  dataset_size: int,
                  device: torch.device,
                  logger: logging.Logger):
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

    net = DevModel(backbone, num_attrs=num_attrs, num_classes=num_classes,
                   use_sigmoid=cfg.MODEL.USE_SIGMOID, use_attention=cfg.MODEL.USE_ATTENTION)
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

    logger.info("Start full intervention evaluation...")
    test_interventions_full(model=net, dataloader=dataloader_test, num_corrects_fn=compute_corrects,
                            dataset_size=len(dataset_test), logger=logger, writer=summary_writer, device=device)

    # Test Intervention Performance
    logger.info("Start intervention evaluation...")
    train_activations = None
    if not cfg.MODEL.USE_SIGMOID:
        logger.info("Model does not use sigmoid activation, generate activations for intervention...")
        train_activations = get_lo_hi_activations(net, dataloader_test, device=device)

    num_groups_to_intervene = [4, 8, 12, 16, 20, 24, 28]
    test_interventions(model=net, dataloader=dataloader_test, num_int_groups=num_groups_to_intervene,
                       attribute_group_indices=dataset_train.attribute_group_indices,
                       use_sigmoid=cfg.MODEL.USE_SIGMOID, train_activations=train_activations,
                       num_corrects_fn=compute_corrects, dataset_size=len(dataset_test), rng=rng,
                       logger=logger, writer=summary_writer, device=device)

    logger.info("DONE!")


if __name__ == "__main__":
    main()
