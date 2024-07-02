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
from models.utils import PPNet, PPNetLoss


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
        logits, l2_dists = model(batch_inputs["pixel_values"])
        loss_dict = loss_fn((logits, l2_dists,),
                            batch_inputs,
                            model.prototype_class_identity,
                            model.last_layer.weight)
        total_loss = sum(loss_dict.values())
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        for loss_name, loss in loss_dict.items():
            running_losses[loss_name] += loss * dataloader.batch_size

        mca(logits, batch_inputs["class_ids"])

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
        logits, l2_dists = model(batch_inputs["pixel_values"])

        mca(logits, batch_inputs["class_ids"])

    epoch_acc = mca.compute().item()
    writer.add_scalar(f"Acc/{model_name}/val", epoch_acc, epoch)
    logger.info(f"EPOCH {epoch} {model_name} Val Acc: {epoch_acc:.4f}")

    return epoch_acc


def main():
    parser = argparse.ArgumentParser(description="Training Script For ProtoPNet")
    args = parser.parse_args()
    seed = 42

    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    experiment_name = "ProtoPNet Experiment"

    print("Experiment Name:", experiment_name)
    print("Hyperparameters:")
    print("Device:", device)

    #################
    # Setup logging #
    #################

    log_dir = Path("logs") / f"CUB_runs" / experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)

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

    use_augmentation = True
    batch_size = 80

    train_transforms, test_transforms = get_transforms_dev(cropped=use_augmentation)
    dataset_train = CUBDataset(Path("datasets") / "CUB", split="train_val",
                               use_attrs="continuous", use_attr_mask=None,
                               use_splits="data/cub/split_image_ids.npz", use_augmentation="augment",
                               transforms=train_transforms, use_parts="coarse")
    print("Training set size:", len(dataset_train))
    dataset_val = CUBDataset(Path("datasets") / "CUB", split="test",
                             use_attrs="continuous", use_attr_mask=None,
                             use_splits="data/cub/split_image_ids.npz", use_augmentation="augment",
                             transforms=train_transforms, use_parts="coarse")
    print("Validation set size:", len(dataset_val))
    dataloader_train = DataLoader(dataset=dataset_train, batch_size=batch_size,
                                  shuffle=True, num_workers=8)
    dataloader_val = DataLoader(dataset=dataset_val, batch_size=batch_size,
                                shuffle=True, num_workers=8)

    #################################
    # Load and fine-tune full model #
    #################################
    from torchvision.models import densenet121, DenseNet121_Weights
    # TODO: Figure out what modification need to be done for backbone
    backbone = densenet121(weights=DenseNet121_Weights.DEFAULT)
    ppnet = PPNet(backbone=backbone, prototype_shape=(2000, 128, 1, 1,),
                  num_classes=200, activation_fn="log", init_weights=True).to(device)

    criterion = PPNetLoss(l_clst_coef=0.8, l_sep_coef=-0.08, l_l1_coef=1e-4)

    # Initialize optimizer
    joint_optimizer_specs = [{'params': ppnet.features.parameters(),
                              'lr': 1e-4,
                              'weight_decay': 1e-3},
                             {'params': ppnet.add_on_layers.parameters(),
                              'lr': 3e-3,
                              'weight_decay': 1e-3},
                             {'params': ppnet.prototype_vectors,
                              'lr': 3e-3}]
    joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
    joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=5, gamma=0.1)

    warm_optimizer_specs = [{'params': ppnet.add_on_layers.parameters(),
                             'lr': 3e-3,
                             'weight_decay': 1e-3},
                            {'params': ppnet.prototype_vectors,
                             'lr': 3e-3}]
    warm_optimizer = torch.optim.Adam(warm_optimizer_specs)

    last_layer_optimizer_specs = [{'params': ppnet.last_layer.parameters(), 'lr': 1e-4}]
    last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

    num_warm_epochs = 5

    ppnet.to(device)
    ppnet.train()
    best_epoch, best_val_acc = 0, 0.
    for epoch in range(31):
        if epoch <= num_warm_epochs:
            for p in ppnet.features.parameters():
                p.requires_grad = False
            train_epoch(model=ppnet, loss_fn=criterion, dataloader=dataloader_train, optimizer=warm_optimizer,
                        writer=summary_writer, epoch=epoch, device=device, logger=logger)
        else:
            for p in ppnet.features.parameters():
                p.requires_grad = True
            train_epoch(model=ppnet, loss_fn=criterion, dataloader=dataloader_train, optimizer=joint_optimizer,
                        writer=summary_writer, epoch=epoch, device=device, logger=logger)
            joint_lr_scheduler.step()
        val_acc = val_epoch(model=ppnet, dataloader=dataloader_val, writer=summary_writer,
                            device=device, epoch=epoch, logger=logger)

        # Early stopping based on validation accuracy
        # if val_acc > best_val_acc:
        #     torch.save({k: v.cpu() for k, v in net.state_dict().items()},
        #                Path(log_dir) / f"{cfg.MODEL.NAME}.pth")
        #     best_val_acc = val_acc
        #     best_epoch = epoch
        #     logger.info("Best epoch found, model saved!")
        # if epoch >= best_epoch + early_stopping_epochs:
        #     break
        #

if __name__ == "__main__":
    main()
