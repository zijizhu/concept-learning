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


@torch.inference_mode()
def test_interventions(model: nn.Module, dataloader: DataLoader, num_int_groups_list: list[int],
                       attribute_group_indices: np.array, num_classes: int, rng: np.random.Generator,
                       logger: logging.Logger, writer: SummaryWriter, device: torch.device):
    """Given a dataset and concept learning model, test its ability of responding to interventions"""
    num_total_groups = len(np.unique(attribute_group_indices))

    for num_int_groups in num_int_groups_list:
        if num_int_groups > num_total_groups:
            continue
        sampled_group_ids, int_masks = [], []
        for _ in range(len(dataloader.dataset)):
            group_id_choices = rng.choice(np.arange(num_total_groups), size=num_int_groups, replace=False)
            mask = np.isin(attribute_group_indices, group_id_choices).astype(int)
            sampled_group_ids.append(group_id_choices)
            int_masks.append(mask)

        int_dataset = TensorDataset(torch.tensor(np.stack(sampled_group_ids)),
                                    torch.tensor(np.stack(int_masks)))
        int_dataloader = DataLoader(int_dataset, batch_size=dataloader.batch_size)

        mca = MulticlassAccuracy(num_classes=num_classes).to(device)
        # Inference loop
        for batch_inputs, batch_int in tqdm(zip(dataloader, int_dataloader), total=len(dataloader)):
            _, int_masks = batch_int
            batch = {k: v.to(device) for k, v in batch_inputs.items()}
            int_masks = int_masks.to(device)
            int_values = batch["attr_scores"]
            outputs = model.inference(batch["pixel_values"], int_mask=int_masks, int_values=int_values)

            mca(outputs["class_preds"], batch_inputs["class_ids"])

        # Compute accuracy
        acc = mca.compute().item()
        writer.add_scalar("Num intervened attributes vs Accuracy", acc, num_int_groups)
        logger.info(f"Test accuracy when {num_int_groups} attribute groups intervened: {acc:.4f}")

        del mca


@torch.inference_mode()
def test_interventions_full(model: nn.Module, dataloader: DataLoader, num_classes: int,
                            logger: logging.Logger, writer: SummaryWriter, device: torch.device):
    """Given a dataset and concept learning model, test its ability of responding to interventions"""
    mca = MulticlassAccuracy(num_classes=num_classes).to(device)

    # Inference loop
    for batch_inputs in tqdm(dataloader):
        batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
        class_preds = model.c2y(batch_inputs["attr_scores"].to(torch.float32))

        mca(class_preds, batch_inputs["class_ids"])

    # Compute accuracy
    acc = mca.compute().item()
    logger.info(f"Test accuracy with Full Intervention: {acc:.4f}")
    writer.add_text(f"Test accuracy with Full Intervention: {acc:.4f}")


@torch.no_grad()
def test_accuracy(model: nn.Module,
                  num_classes: int,
                  dataloader: DataLoader,
                  device: torch.device,
                  writer: SummaryWriter,
                  logger: logging.Logger):
    mca = MulticlassAccuracy(num_classes=num_classes).to(device)

    for batch_inputs in tqdm(dataloader):
        batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
        outputs = model(batch_inputs['pixel_values'])

        mca(outputs["class_preds"], batch_inputs["class_ids"])

    acc = mca.compute().item()
    logger.info(f"Test accuracy: {acc:.4f}")
    writer.add_text(f"Test accuracy: {acc:.4f}")


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

    if cfg.DATASET.NAME == "CUB":
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
    else:
        raise NotImplementedError

    ###############
    # Load models #
    ###############

    if cfg.MODEL.BACKBONE == 'resnet101':
        from torchvision.models import ResNet101_Weights, resnet101
        backbone = resnet101(weights=ResNet101_Weights.DEFAULT)
    elif cfg.MODEL.BACKBONE == 'resnet50':
        from torchvision.models import ResNet50_Weights, resnet50
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
    
    if "accuracy" in args.metrics:
        logger.info("Start task accuracy evaluation...")
        test_accuracy(model=net, num_classes=num_classes, dataloader=dataloader_test,
                    device=device, writer=summary_writer, logger=logger)
    if "full_int_acc" in args.metrics:
        logger.info("Start full intervention evaluation...")
        test_interventions_full(model=net, dataloader=dataloader_test, num_classes=num_classes,
                                logger=logger, writer=summary_writer, device=device)

    # Test Intervention Performance
    if "int_acc_curve" in args.metrics:
        logger.info("Start intervention evaluation for different number of attribute groups intervened...")
        num_groups_to_intervene = [4, 8, 12, 16, 20, 24, 28]
        test_interventions(model=net, dataloader=dataloader_test,
                           num_int_groups_list=num_groups_to_intervene,
                           attribute_group_indices=dataset_test.attribute_group_indices,
                           num_classes=num_classes, rng=rng, logger=logger, writer=summary_writer,
                           device=device)

    # Test Attribute and Part Localization Performance
    if "part_iou" in args.metrics:
        dataset_test_no_transform = CUBDataset(Path(cfg.DATASET.ROOT_DIR) / "CUB", split="test",
                                               use_attrs=cfg.DATASET.USE_ATTRS,
                                               use_attr_mask=cfg.DATASET.USE_ATTR_MASK,
                                               use_splits=cfg.DATASET.USE_SPLITS,
                                               use_augmentation=augmentation, transforms=None)

        loc_eval(net, dataset_test_no_transform, log_dir, logger,
                cropped=bool(augmentation), bbox_size=90, device=device)

    summary_writer.flush()
    summary_writer.close()
    logger.info("DONE!")


if __name__ == "__main__":
    main()
