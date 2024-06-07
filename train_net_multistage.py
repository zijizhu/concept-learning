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
from data.cub.transforms import get_transforms_cbm
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
    writer.add_scalar("Acc/{model_name}/train", epoch_acc, epoch)
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

    experiment_name = config_path.stem
    print("Experiment Name:", experiment_name)
    print("Hyperparameters:")
    print(OmegaConf.to_yaml(cfg))
    print("Device:", device)

    #################
    # Setup logging #
    #################
    log_dir = Path("logs") / "CUB_runs" / f"dev_{cfg.MODEL.LOSSES.L_C}_{cfg.OPTIM.LR}"
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(os.path.join(log_dir, "hparams.yaml"), "w+") as fp:
        OmegaConf.save(OmegaConf.merge(OmegaConf.create({"NAME": experiment_name}), cfg), f=fp.name)

    summary_writer = SummaryWriter(log_dir=str(log_dir))
    summary_writer.add_text("Model", cfg.MODEL.NAME)
    summary_writer.add_text("Dataset", cfg.DATASET.NAME)
    summary_writer.add_text("Batch size", str(cfg.OPTIM.BATCH_SIZE))
    summary_writer.add_text("Epochs", str(cfg.OPTIM.EPOCHS))
    summary_writer.add_text("Seed", str(cfg.SEED))

    summary_writer.add_text("Attribute Loss Coefficient", str(cfg.MODEL.LOSSES.L_C))

    summary_writer.add_text("Main Model Learning Rate", str(cfg.OPTIM.LR))
    summary_writer.add_text("Backbone Finetuning Learning Rate", str(cfg.OPTIM.LR_BACKBONE))
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
        train_transforms, test_transforms = get_transforms_cbm()

        num_attrs = cfg.get("DATASET.NUM_ATTRS", 112)
        num_classes = 200
        dataset_train = CUBDataset(
            Path(cfg.DATASET.ROOT_DIR) / "CUB", split="train", use_attrs=cfg.DATASET.USE_ATTRS,
            use_attr_mask=cfg.DATASET.USE_ATTR_MASK, use_splits=cfg.DATASET.USE_SPLITS,
            transforms=train_transforms)
        dataset_val = CUBDataset(
            Path(cfg.DATASET.ROOT_DIR) / "CUB", split="val", use_attrs=cfg.DATASET.USE_ATTRS,
            use_attr_mask=cfg.DATASET.USE_ATTR_MASK, use_splits=cfg.DATASET.USE_SPLITS,
            transforms=train_transforms)
        dataloader_train = DataLoader(
            dataset=dataset_train, batch_size=cfg.OPTIM.BATCH_SIZE,
            shuffle=True, num_workers=8)
        dataloader_val = DataLoader(
            dataset=dataset_val, batch_size=cfg.OPTIM.BATCH_SIZE,
            shuffle=True, num_workers=8)
    else:
        raise NotImplementedError

    ###############################
    # Load and fine-tune backbone #
    ###############################

    if cfg.MODEL.BACKBONE == 'resnet101':
        from torchvision.models import resnet101, ResNet101_Weights
        backbone = resnet101(weights=ResNet101_Weights.DEFAULT)
    elif cfg.MODEL.BACKBONE == 'resnet50':
        from torchvision.models import resnet50, ResNet50_Weights
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
    else:
        raise NotImplementedError
    backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
    optimizer = optim.Adam(backbone.parameters(), lr=cfg.OPTIM.LR_BACKBONE)

    def criterion(outputs, batch_inputs):
        loss = f.cross_entropy(outputs, batch_inputs["class_ids"])
        return {'l_y': loss}, loss

    logger.info("Start tuning backbone...")
    backbone.to(device)
    backbone.train()
    best_epoch, best_val_acc = 0, 0.
    for epoch in range(cfg.OPTIM.EPOCHS):
        train_epoch(model=backbone, loss_fn=criterion, loss_keys=['l_y'], num_corrects_fn=compute_corrects,
                    dataloader=dataloader_train, optimizer=optimizer, writer=summary_writer,
                    batch_size=cfg.OPTIM.BATCH_SIZE, dataset_size=len(dataset_train), device=device,
                    epoch=epoch, logger=logger, model_name="backbone")

        val_acc = val_epoch(model=backbone, num_corrects_fn=compute_corrects, dataloader=dataloader_val,
                            writer=summary_writer, dataset_size=len(dataset_val), device=device,
                            epoch=epoch, logger=logger, model_name="backbone")

        # Early stopping based on validation accuracy
        if val_acc > best_val_acc:
            torch.save({k: v.cpu() for k, v in backbone.state_dict().items()},
                       Path(log_dir) / f"{cfg.MODEL.BACKBONE}.pt")
            best_val_acc = val_acc
            best_epoch = epoch
        if epoch >= best_epoch + 10:
            break

    print()

    #################################
    # Load and fine-tune full model #
    #################################
    state_dict = torch.load(Path(log_dir) / f"{cfg.MODEL.BACKBONE}.pt")
    backbone.load_state_dict(state_dict)

    net = DevModel(backbone, num_attrs=num_attrs, num_classes=num_classes, activation=cfg.MODEL.ACTIVATION)

    loss_coef_dict = {k.lower(): v for k, v in dict(cfg.MODEL.LOSSES).items()}
    criterion = DevLoss(torch.tensor(dataset_train.attribute_weights), device=device, **loss_coef_dict)

    optimizer = optim.AdamW(params=[
        {"params": net.backbone.parameters(), "lr": cfg.OPTIM.LR * 0.1},
        {"params": net.prototype_conv.parameters()},
        {"params": net.c2y.parameters()}
    ], lr=cfg.OPTIM.LR, weight_decay=cfg.OPTIM.WEIGHT_DECAY)

    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.OPTIM.STEP_SIZE, gamma=cfg.OPTIM.GAMMA)
    scheduler = None  # type: optim.lr_scheduler.StepLR | None

    for param in net.backbone.parameters():
        param.requires_grad = False

    logger.info("Start training prototypes and classification layers...")

    net.to(device)
    net.train()
    best_epoch, best_val_acc = 0, 0.
    prototype_weights = []
    for epoch in range(cfg.OPTIM.EPOCHS):
        if epoch == 10:
            for param in net.backbone.parameters():
                param.requires_grad = False
        train_epoch(model=net, loss_fn=criterion, loss_keys=list(loss_coef_dict.keys()),
                    num_corrects_fn=compute_corrects, dataloader=dataloader_train, optimizer=optimizer,
                    writer=summary_writer, batch_size=cfg.OPTIM.BATCH_SIZE, dataset_size=len(dataset_train),
                    device=device, epoch=epoch, logger=logger, model_name="full model")

        val_acc = val_epoch(model=net, num_corrects_fn=compute_corrects, dataloader=dataloader_val,
                            writer=summary_writer, dataset_size=len(dataset_val), device=device,
                            epoch=epoch, logger=logger, model_name="full model")
        if scheduler:
            scheduler.step()

        # Early stopping based on validation accuracy
        if val_acc > best_val_acc:
            torch.save({k: v.cpu() for k, v in backbone.state_dict().items()},
                       Path(log_dir) / f"{cfg.MODEL.NAME}.pt")
            best_val_acc = val_acc
            best_epoch = epoch
        if epoch >= best_epoch + 30:
            break

        # Save prototype weights for inspection
        prototype_weights.append(net.prototype_conv.weight.detach().cpu())

    torch.save(torch.stack(prototype_weights), Path(log_dir) / "prototype_weights.pt")


if __name__ == "__main__":
    main()