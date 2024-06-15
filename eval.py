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
from torch.utils.data import DataLoader, TensorDataset
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
def test_interventions(model: nn.Module, dataloader: DataLoader, num_int_groups_list: list[int],
                       attribute_group_indices: np.array, num_corrects_fn: Callable,
                       dataset_size: int, batch_size: int, rng: np.random.Generator, logger: logging.Logger,
                       writer: SummaryWriter, device: torch.device):
    """Given a dataset and concept learning model, test its ability of responding to test-time interventions"""
    num_total_groups = len(np.unique(attribute_group_indices))

    for num_int_groups in num_int_groups_list:
        sampled_group_ids, int_masks = [], []
        for _ in range(dataset_size):
            group_id_choices = rng.choice(np.arange(num_total_groups), size=num_int_groups, replace=False)
            mask = np.isin(attribute_group_indices, group_id_choices).astype(int)
            sampled_group_ids.append(group_id_choices)
            int_masks.append(mask)

        int_dataset = TensorDataset(torch.tensor(np.stack(sampled_group_ids)),
                                    torch.tensor(np.stack(int_masks)))
        int_dataloader = DataLoader(int_dataset, batch_size=batch_size)

        running_corrects = 0
        # Inference loop
        for batch, int_batch in tqdm(zip(dataloader, int_dataloader), total=len(dataloader)):
            _, int_masks = int_batch
            batch = {k: v.to(device) for k, v in batch.items()}
            int_masks = int_masks.to(device)
            int_values = batch["attr_scores"]
            results = model.inference(batch["pixel_values"], int_mask=int_masks, int_values=int_values)

            running_corrects += num_corrects_fn(results, batch)

        # Compute accuracy
        acc = running_corrects / dataset_size
        writer.add_scalar("Acc/intervention curve", acc, num_int_groups)
        logger.info(f"Test Acc when {num_int_groups} attribute groups intervened: {acc:.4f}")


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
        dataset_train = CUBDataset(Path(cfg.DATASET.ROOT_DIR) / "CUB", split="train_val",
                                   use_attrs=cfg.DATASET.USE_ATTRS, use_attr_mask=cfg.DATASET.USE_ATTR_MASK,
                                   use_splits=cfg.DATASET.USE_SPLITS, transforms=train_transforms)
        dataset_test = CUBDataset(Path(cfg.DATASET.ROOT_DIR) / "CUB", split="test",
                                  use_attrs=cfg.DATASET.USE_ATTRS, use_attr_mask=cfg.DATASET.USE_ATTR_MASK,
                                  use_splits=cfg.DATASET.USE_SPLITS, transforms=train_transforms)
        dataloader_train = DataLoader(dataset=dataset_train, batch_size=cfg.OPTIM.BATCH_SIZE,
                                      shuffle=True, num_workers=8)
        dataloader_test = DataLoader(dataset=dataset_test, batch_size=cfg.OPTIM.BATCH_SIZE,
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
                   use_attention=cfg.MODEL.USE_ATTENTION)
    state_dict = torch.load(log_dir / f"{cfg.MODEL.NAME}.pth", map_location=device)
    net.load_state_dict(state_dict)

    ###############
    # Evaluations #
    ###############

    net.to(device)
    net.eval()

    # Test Accuracy
    logger.info("Start task accuracy evaluation...")
    test_accuracy(net, compute_corrects, dataloader_test, len(dataset_test), device, logger)

    logger.info("Start full intervention evaluation...")
    test_interventions_full(model=net, dataloader=dataloader_test, num_corrects_fn=compute_corrects,
                            dataset_size=len(dataset_test), logger=logger, writer=summary_writer, device=device)

    # Test Intervention Performance
    logger.info("Start intervention evaluation...")

    num_groups_to_intervene = [4, 8, 12, 16, 20, 24, 28]
    test_interventions(model=net, dataloader=dataloader_test, num_int_groups_list=num_groups_to_intervene,
                       attribute_group_indices=dataset_train.attribute_group_indices,
                       batch_size=cfg.OPTIM.BATCH_SIZE, num_corrects_fn=compute_corrects,
                       dataset_size=len(dataset_test), rng=rng, logger=logger, writer=summary_writer, device=device)

    summary_writer.flush()
    summary_writer.close()
    logger.info("DONE!")


if __name__ == "__main__":
    main()
