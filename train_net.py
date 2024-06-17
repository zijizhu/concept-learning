import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as f
from lightning import seed_everything
from omegaconf import OmegaConf
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from data.cub.cub_dataset import CUBDataset
from data.cub.transforms import get_transforms_dev
from models.dev import DevModel, DevLoss


# TODO Move this function to other files
@torch.no_grad()
def compute_corrects(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]):
    if isinstance(outputs, dict):
        class_preds = outputs["class_scores"]
    else:
        class_preds = outputs
    class_ids = batch["class_ids"]
    return torch.sum(torch.argmax(class_preds.data, dim=-1) == class_ids.data).item()


def train_epoch(model: nn.Module,
                loss_fn: nn.Module | Callable,
                loss_keys: list[str],
                num_corrects_fn: nn.Module | Callable,
                dataloader: DataLoader,
                optimizer: torch.optim.Optimizer,
                writer: SummaryWriter,
                dataset_size: int,
                epoch: int,
                batch_size: int,
                device: torch.device,
                logger: logging.Logger,
                model_name: str):
    running_losses = {k: 0 for k in loss_keys}
    running_corrects = 0

    for batch_inputs in tqdm(dataloader):
        batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
        outputs = model(batch_inputs["pixel_values"])
        loss_dict, total_loss = loss_fn(outputs, batch_inputs)

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        for loss_name, loss in loss_dict.items():
            running_losses[loss_name] += loss * batch_size

        running_corrects += num_corrects_fn(outputs, batch_inputs)

    # Log metrics
    for loss_name, loss in running_losses.items():
        loss_avg = loss / dataset_size
        writer.add_scalar(f"Loss/{model_name}/train/{loss_name}", loss_avg, epoch)
        logger.info(f"EPOCH {epoch} {model_name} Train {loss_name}: {loss_avg:.4f}")

    epoch_acc = running_corrects / dataset_size
    writer.add_scalar(f"Acc/{model_name}/train", epoch_acc, epoch)
    logger.info(f"EPOCH {epoch} {model_name} Train Acc: {epoch_acc:.4f}")


@torch.no_grad()
def val_epoch(model: nn.Module,
              num_corrects_fn: nn.Module | Callable,
              dataloader: DataLoader,
              writer: SummaryWriter,
              dataset_size: int,
              epoch: int,
              device: torch.device,
              logger: logging.Logger,
              model_name: str):
    running_corrects = 0
    for batch_inputs in tqdm(dataloader):
        batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
        outputs = model(batch_inputs["pixel_values"])

        running_corrects += num_corrects_fn(outputs, batch_inputs)

    epoch_acc = running_corrects / dataset_size
    writer.add_scalar(f"Acc/{model_name}/val", epoch_acc, epoch)
    logger.info(f"EPOCH {epoch} {model_name} Val Acc: {epoch_acc:.4f}")

    return epoch_acc


def main():
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument("-c", "--config_path", type=str, required=True)
    parser.add_argument("-n", "--name", type=str, required=False)
    parser.add_argument("-o", "--options", type=str, nargs="+")

    args = parser.parse_args()
    config_path = Path(args.config_path)
    base_cfg = OmegaConf.load(config_path)
    if args.options:
        cli_cfg = OmegaConf.from_dotlist(args.options)
        cfg = OmegaConf.merge(base_cfg, cli_cfg)
    else:
        cfg = base_cfg

    seed_everything(cfg.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.name:
        experiment_name = args.name
    else:
        experiment_name = f"{config_path.stem}_base"

    print("Experiment Name:", experiment_name)
    print("Hyperparameters:")
    print(OmegaConf.to_yaml(cfg))
    print("Device:", device)

    #################
    # Setup logging #
    #################

    log_dir = Path("logs") / f"{cfg.DATASET.NAME}_runs" / experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(os.path.join(log_dir, "hparams.yaml"), "w+") as fp:
        OmegaConf.save(cfg, f=fp.name)

    summary_writer = SummaryWriter(log_dir=str(log_dir))
    summary_writer.add_text("Model", cfg.MODEL.NAME)
    summary_writer.add_text("Dataset", cfg.DATASET.NAME)
    summary_writer.add_text("Batch size", str(cfg.OPTIM.BATCH_SIZE))
    summary_writer.add_text("Epochs", str(cfg.OPTIM.EPOCHS))
    summary_writer.add_text("Seed", str(cfg.SEED))

    summary_writer.add_text("Attribute Loss Coefficient", str(cfg.MODEL.LOSSES.L_C))
    summary_writer.add_text("Attention Map Loss Coefficient", str(cfg.MODEL.LOSSES.L_CPT))

    summary_writer.add_text("Main Model Learning Rate", str(cfg.OPTIM.LR))
    summary_writer.add_text("Backbone Finetuning", str(cfg.OPTIM.get("BACKBONE_FT", "N/A")))
    summary_writer.add_text("Weight Decay", str(cfg.OPTIM.WEIGHT_DECAY))


    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "train.log")),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger(__name__)

    #################################
    # Setup datasets and transforms #
    #################################

    if cfg.DATASET.NAME == "CUB":
        augmentation = cfg.DATASET.get("AUGMENTATION", None)
        train_transforms, test_transforms = get_transforms_dev(cropped=True if augmentation else False)
        num_classes = 200
        num_attrs = cfg.DATASET.get("NUM_ATTRS", 112)
        dataset_train = CUBDataset(Path(cfg.DATASET.ROOT_DIR) / "CUB", split="train_val",
                                   use_attrs=cfg.DATASET.USE_ATTRS, use_attr_mask=cfg.DATASET.USE_ATTR_MASK,
                                   use_splits=cfg.DATASET.USE_SPLITS, use_augmentation=augmentation,
                                   transforms=train_transforms)
        print("Training set size:", len(dataset_train))
        dataset_val = CUBDataset(Path(cfg.DATASET.ROOT_DIR) / "CUB", split="test",
                                 use_attrs=cfg.DATASET.USE_ATTRS, use_attr_mask=cfg.DATASET.USE_ATTR_MASK,
                                 use_splits=cfg.DATASET.USE_SPLITS, use_augmentation=augmentation,
                                 transforms=train_transforms)
        print("Validation set size:", len(dataset_val))
        dataloader_train = DataLoader(dataset=dataset_train, batch_size=cfg.OPTIM.BATCH_SIZE,
                                      shuffle=True, num_workers=8)
        dataloader_val = DataLoader(dataset=dataset_val, batch_size=cfg.OPTIM.BATCH_SIZE, shuffle=True, num_workers=8)
    else:
        raise NotImplementedError

    #################################
    # Load and fine-tune full model #
    #################################
    if cfg.MODEL.BACKBONE == 'resnet101':
        from torchvision.models import resnet101, ResNet101_Weights
        backbone = resnet101(weights=ResNet101_Weights.DEFAULT)
    elif cfg.MODEL.BACKBONE == 'resnet50':
        from torchvision.models import resnet50, ResNet50_Weights
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
    else:
        raise NotImplementedError

    net = DevModel(backbone, num_attrs=num_attrs, num_classes=num_classes, use_attention=cfg.MODEL.USE_ATTENTION)
    loss_keys = ['l_y', 'l_c'] + (['l_cpt'] if cfg.MODEL.LOSSES.L_CPT > 0 else [])
    if cfg.MODEL.LOSSES.USE_ATTR_WEIGHTS:
        attribute_weights = torch.tensor(dataset_train.attribute_weights, device=device)
    else:
        attribute_weights = None
    criterion = DevLoss(l_c_coef=cfg.MODEL.LOSSES.L_C,
                        l_y_coef=cfg.MODEL.LOSSES.L_Y,
                        l_cpt_coef=cfg.MODEL.LOSSES.L_CPT,
                        attribute_weights=attribute_weights)

    # Initialize optimizer
    non_backbone_params = [p for name, p in net.named_parameters() if 'backbone' not in name]
    if cfg.OPTIM.OPTIMIZER == "ADAM":
        optimizer = optim.Adam(params=[
            {"params": net.backbone.parameters(), "lr": cfg.OPTIM.LR * 0.1},
            {"params": non_backbone_params},
        ], lr=cfg.OPTIM.LR, weight_decay=cfg.OPTIM.WEIGHT_DECAY)
    elif cfg.OPTIM.OPTIMIZER == "SGD":
        optimizer = optim.SGD(params=[
            {"params": net.backbone.parameters(), "lr": cfg.OPTIM.LR * 0.1},
            {"params": non_backbone_params},
        ], lr=cfg.OPTIM.LR, weight_decay=cfg.OPTIM.WEIGHT_DECAY, momentum=0.9)
    else:
        raise NotImplementedError

    net.to(device)
    net.train()
    best_epoch, best_val_acc = 0, 0.
    early_stopping_epochs = cfg.OPTIM.get("EARLY_STOP", 30)
    prototype_weights = []
    for epoch in range(cfg.OPTIM.EPOCHS):
        train_epoch(model=net, loss_fn=criterion, loss_keys=loss_keys, num_corrects_fn=compute_corrects,
                    dataloader=dataloader_train, optimizer=optimizer, writer=summary_writer,
                    batch_size=cfg.OPTIM.BATCH_SIZE, dataset_size=len(dataset_train),
                    device=device, epoch=epoch, logger=logger, model_name="full model")

        val_acc = val_epoch(model=net, num_corrects_fn=compute_corrects, dataloader=dataloader_val,
                            writer=summary_writer, dataset_size=len(dataset_val), device=device,
                            epoch=epoch, logger=logger, model_name="full model")

        # Early stopping based on validation accuracy
        if val_acc > best_val_acc:
            torch.save({k: v.cpu() for k, v in net.state_dict().items()},
                       Path(log_dir) / f"{cfg.MODEL.NAME}.pth")
            best_val_acc = val_acc
            best_epoch = epoch
        if epoch >= best_epoch + early_stopping_epochs:
            break

        # Save prototype weights for inspection
        prototype_weights.append(net.prototype_conv.weight.detach().cpu())
        torch.save(torch.stack(prototype_weights, dim=0), Path(log_dir) / "prototype_weights.pth")


if __name__ == "__main__":
    main()
