import argparse
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable

import torch
import torchvision.transforms as T
from lightning import seed_everything
from omegaconf import OmegaConf
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchmetrics.classification import MulticlassAccuracy, MultilabelAccuracy
from tqdm import tqdm

from data.cub.cub_dataset import CUBDataset
from models.dino import DINOPPNet, DINOPPNetLoss


def train_epoch(model: nn.Module,
                loss_fn: nn.Module | Callable,
                dataloader: DataLoader,
                optimizer: optim.Optimizer,
                writer: SummaryWriter,
                epoch: int,
                device: torch.device,
                logger: logging.Logger):
    model_name = type(model).__name__
    running_losses = defaultdict(float)
    # mca = MulticlassAccuracy(num_classes=200).to(device)
    mla = MultilabelAccuracy(num_labels=112).to(device)

    for batch_inputs in tqdm(dataloader):
        batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
        outputs = model(batch_inputs["pixel_values"])
        loss_dict = loss_fn(outputs, batch_inputs, model.prototype_class_identity)
        total_loss = sum(loss_dict.values())

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        for loss_name, loss in loss_dict.items():
            running_losses[loss_name] += loss * dataloader.batch_size
        
        # mca(outputs["class_preds"], batch_inputs["class_ids"])
        mla(outputs["attr_preds"], batch_inputs["attr_scores"])

    # Log metrics
    for loss_name, loss in running_losses.items():
        loss_avg = loss / len(dataloader.dataset)
        writer.add_scalar(f"Loss/{model_name}/train/{loss_name}", loss_avg, epoch)
        logger.info(f"EPOCH {epoch} {model_name} Train {loss_name}: {loss_avg:.4f}")

    epoch_acc = mla.compute().item()
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
    # mca = MulticlassAccuracy(num_classes=200).to(device)
    mla = MultilabelAccuracy(num_labels=112).to(device)

    for batch_inputs in tqdm(dataloader):
        batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
        outputs = model(batch_inputs["pixel_values"])

        mla(outputs["attr_preds"], batch_inputs["attr_scores"])

    epoch_acc = mla.compute().item()
    writer.add_scalar(f"Acc/{model_name}/val", epoch_acc, epoch)
    logger.info(f"EPOCH {epoch} {model_name} Val Acc: {epoch_acc:.4f}")

    return epoch_acc


def main():
    parser = argparse.ArgumentParser(description="DINO Training Script")
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

    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    experiment_name = args.name if args.name else f"{config_path.stem}_base"

    print("Experiment Name:", experiment_name)
    print("Hyperparameters:")
    print(OmegaConf.to_yaml(cfg))
    print("Device:", device)

    #################
    # Setup logging #
    #################

    log_dir = Path("logs") / f"{cfg.dataset.name}_runs" / experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(os.path.join(log_dir, "hparams.yaml"), "w+") as fp:
        OmegaConf.save(cfg, f=fp.name)

    summary_writer = SummaryWriter(log_dir=str(log_dir))

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

    augmentation = cfg.dataset.get("augmentation", None)
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406,)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225,)
    transforms = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    ])
    dataset_train = CUBDataset(Path(cfg.dataset.root_dir) / "CUB", split="train_val",
                                use_attrs=cfg.dataset.use_attrs, use_attr_mask=cfg.dataset.use_attr_mask,
                                use_splits=cfg.dataset.use_splits, use_augmentation=augmentation,
                                transforms=transforms, use_parts="coarse")
    print("Training set size:", len(dataset_train))
    dataset_val = CUBDataset(Path(cfg.dataset.root_dir) / "CUB", split="test",
                                use_attrs=cfg.dataset.use_attrs, use_attr_mask=cfg.dataset.use_attr_mask,
                                use_splits=cfg.dataset.use_splits, use_augmentation=augmentation,
                                transforms=transforms, use_parts="coarse")
    print("Validation set size:", len(dataset_val))
    dataloader_train = DataLoader(dataset=dataset_train, batch_size=cfg.optim.batch_size,
                                    shuffle=True, num_workers=8)
    dataloader_val = DataLoader(dataset=dataset_val, batch_size=cfg.optim.batch_size,
                                shuffle=True, num_workers=8)

    #################################
    # Load and fine-tune full model #
    #################################
    num_attrs = cfg.dataset.num_attrs
    K = 5
    backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
    net = DINOPPNet(backbone=backbone, prototype_shape=(num_attrs * K, 768), num_classes=num_attrs,
                    init_weights=True, activation_fn="log", use_relu=True)
    criterion = DINOPPNetLoss(l_c_coef=cfg.model.loss.l_c,
                              l_clst_coef=cfg.model.loss.l_clst,
                              l_sep_coef=cfg.model.loss.l_sep,
                              k=K).to(device=device)

    # Initialize optimizer
    for name, param in net.named_parameters():
        if "backbone" in name:
            param.requires_grad = False
    optim_args = dict(params=filter(lambda p: p.requires_grad, net.parameters()),
                      lr=cfg.optim.lr)
    optimizer = optim.AdamW(**optim_args, weight_decay=cfg.optim.weight_decay)

    net.to(device)
    net.train()
    best_epoch, best_val_acc = 0, 0.
    early_stopping_epochs = cfg.optim.get("early_stop", 10)

    for epoch in range(cfg.optim.epochs):
        print(f"EPOCH {epoch} learning rate:")
        for param_group in optimizer.param_groups:
            print(param_group['lr'])

        train_epoch(model=net, loss_fn=criterion, dataloader=dataloader_train, optimizer=optimizer,
                    writer=summary_writer, epoch=epoch, device=device, logger=logger)

        val_acc = val_epoch(model=net, dataloader=dataloader_val, writer=summary_writer,
                            device=device, epoch=epoch, logger=logger)

        # Early stopping based on validation accuracy
        if val_acc > best_val_acc:
            torch.save({k: v.cpu() for k, v in net.state_dict().items()},
                       Path(log_dir) / f"{cfg.model.name}.pth")
            best_val_acc = val_acc
            best_epoch = epoch
            logger.info("Best epoch found, model saved!")
        if epoch >= best_epoch + early_stopping_epochs:
            break


if __name__ == "__main__":
    main()
