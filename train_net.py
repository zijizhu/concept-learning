import argparse
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable

import torch
from lightning import seed_everything
from omegaconf import OmegaConf
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm

from data.cub.cub_dataset import CUBDataset
from data.cub.transforms import get_transforms_dev
from models.dev import DevLoss, DevModel


def train_epoch(model: nn.Module,
                loss_fn: nn.Module | Callable,
                dataloader: DataLoader,
                optimizer: torch.optim.Optimizer,
                writer: SummaryWriter,
                epoch: int,
                device: torch.device,
                logger: logging.Logger):
    model_name = type(model).__name__
    running_losses = defaultdict(float)
    mca = MulticlassAccuracy(num_classes=200).to(device)

    for batch_inputs in tqdm(dataloader):
        batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
        outputs = model(batch_inputs["pixel_values"])
        loss_dict, total_loss = loss_fn(outputs, batch_inputs)

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        for loss_name, loss in loss_dict.items():
            running_losses[loss_name] += loss * dataloader.batch_size
        
        mca(outputs["class_preds"], batch_inputs["class_ids"])

    # Log metrics
    for loss_name, loss in running_losses.items():
        loss_avg = loss / len(dataloader.dataset)
        writer.add_scalar(f"Loss/{model_name}/train/{loss_name}", loss_avg, epoch)
        logger.info(f"EPOCH {epoch} {model_name} Train {loss_name}: {loss_avg:.4f}")

    epoch_acc = mca.compute().item()
    writer.add_scalar(f"Acc/{model_name}/train", epoch_acc, epoch)
    logger.info(f"EPOCH {epoch} {model_name} Train Acc: {epoch_acc:.4f}")


@torch.no_grad()
def val_epoch(model: nn.Module,
              dataloader: DataLoader,
              writer: SummaryWriter,
              epoch: int,
              device: torch.device,
              logger: logging.Logger):
    model_name = type(model).__name__
    mca = MulticlassAccuracy(num_classes=200).to(device)

    for batch_inputs in tqdm(dataloader):
        batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
        outputs = model(batch_inputs["pixel_values"])

        mca(outputs["class_preds"], batch_inputs["class_ids"])

    epoch_acc = mca.compute().item()
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

    experiment_name = args.name if args.name else f"{config_path.stem}_base"

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
        force=True,
    )
    logger = logging.getLogger(__name__)

    #################################
    # Setup datasets and transforms #
    #################################

    if cfg.DATASET.NAME == "CUB":
        augmentation = cfg.DATASET.get("AUGMENTATION", None)
        train_transforms, test_transforms = get_transforms_dev(cropped=bool(augmentation))
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
        dataloader_val = DataLoader(dataset=dataset_val, batch_size=cfg.OPTIM.BATCH_SIZE,
                                    shuffle=True, num_workers=8)
    else:
        raise NotImplementedError

    #################################
    # Load and fine-tune full model #
    #################################
    if cfg.MODEL.BACKBONE == 'resnet101':
        from torchvision.models import ResNet101_Weights, resnet101
        backbone = resnet101(weights=ResNet101_Weights.DEFAULT)
    elif cfg.MODEL.BACKBONE == 'resnet50':
        from torchvision.models import ResNet50_Weights, resnet50
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
    else:
        raise NotImplementedError

    net = DevModel(backbone, num_attrs=num_attrs, num_classes=num_classes)
    if cfg.MODEL.LOSSES.USE_ATTR_WEIGHTS:
        attribute_weights = torch.tensor(dataset_train.attribute_weights, device=device)
    else:
        attribute_weights = None

    criterion = DevLoss(l_c_coef=cfg.MODEL.LOSSES.L_C,
                        l_y_coef=cfg.MODEL.LOSSES.L_Y,
                        l_cpt_coef=cfg.MODEL.LOSSES.L_CPT,
                        l_dec_coef=cfg.MODEL.LOSSES.L_DEC,
                        group_indices=dataset_train.part_indices_pt.to(device),
                        attribute_weights=attribute_weights)

    # Initialize optimizer
    optim_args = dict(params=filter(lambda p: p.requires_grad, net.parameters()),
                      lr=cfg.OPTIM.LR, weight_decay=cfg.OPTIM.WEIGHT_DECAY)
    if cfg.OPTIM.OPTIMIZER == "SGD":
        optim_args["momentum"] = 0.9
        optimizer = optim.SGD(**optim_args)
    elif cfg.OPTIM.OPTIMIZER == "ADAM":
        optimizer = optim.Adam(**optim_args)
    else:
        raise NotImplementedError

    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=cfg.OPTIM.STEP_SIZE,
                                          gamma=cfg.OPTIM.GAMMA)

    net.to(device)
    net.train()
    best_epoch, best_val_acc = 0, 0.
    early_stopping_epochs = cfg.OPTIM.get("EARLY_STOP", 30)

    for epoch in range(cfg.OPTIM.EPOCHS):
        train_epoch(model=net, loss_fn=criterion, dataloader=dataloader_train, optimizer=optimizer,
                    writer=summary_writer, epoch=epoch, device=device, logger=logger)

        val_acc = val_epoch(model=net, dataloader=dataloader_val, writer=summary_writer,
                            device=device, epoch=epoch, logger=logger)

        # Early stopping based on validation accuracy
        if val_acc > best_val_acc:
            torch.save({k: v.cpu() for k, v in net.state_dict().items()},
                       Path(log_dir) / f"{cfg.MODEL.NAME}.pth")
            best_val_acc = val_acc
            best_epoch = epoch
            logger.info("Best epoch found, model saved!")
        if epoch >= best_epoch + early_stopping_epochs:
            break

        scheduler.step()


if __name__ == "__main__":
    main()
