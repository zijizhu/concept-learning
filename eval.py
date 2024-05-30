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
from data.cub.transforms import get_transforms_cbm
from models.utils import Backbone
from models.cbm import CBM

@torch.no_grad()
def compute_corrects(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]):
    class_preds, class_ids = outputs["class_preds"], batch["class_ids"]
    return torch.sum(torch.argmax(class_preds.data, dim=-1) == class_ids.data).item()


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
    class_preds, class_ids = outputs["class_preds"], batch["class_ids"]
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
        outputs = model(batch_inputs)

        running_corrects += num_corrects_fn(outputs, batch_inputs)

    epoch_acc = running_corrects / dataset_size
    logger.info(f"Test Acc: {epoch_acc:.4f}")

    return epoch_acc


def main():
    parser = argparse.ArgumentParser(description="Evaluation Script")
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
    rng = np.random.default_rng(cfg.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    experiment_name = config_path.stem
    print("Experiment Name:", experiment_name)
    print("Hyperparameters:")
    print(OmegaConf.to_yaml(cfg))
    print("Device:", device)

    #################
    # Setup logging #
    #################

    log_dir = Path("logs") / f'{datetime.now().strftime("%Y-%m-%d_%H-%M")}_{experiment_name}'
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "hparams.yaml", "w+") as fp:
        OmegaConf.save(OmegaConf.merge(OmegaConf.create({"NAME": experiment_name}), cfg), f=fp.name)

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
    )
    logger = logging.getLogger(__name__)

    #################################
    # Setup datasets and transforms #
    #################################

    if cfg.DATASET.NAME == "CUB":
        if cfg.DATASET.PREPROCESS == "CBM":
            _, test_transforms = get_transforms_cbm()
        else:
            raise NotImplementedError

        num_attrs = cfg.get("DATASET.NUM_ATTRS", 112)
        groups = cfg.get("DATASET.GROUPS", "attributes")
        use_attrs = cfg.get("DATASET.USE_ATTRS", "binary")
        num_classes = 200

        dataset_test = CUBDataset(
            os.path.join(cfg.DATASET.ROOT_DIR, "CUB"),
            use_attrs=use_attrs,
            num_attrs=num_attrs,
            split="test",
            groups=groups,
            transforms=test_transforms,
        )
        dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=4)
    elif cfg.DATASET.NAME == "CELEB":
        raise NotImplementedError
    else:
        raise NotImplementedError

    ###############
    # Load models #
    ###############
    if "CBM" in experiment_name:
        backbone = timm.create_model(cfg.MODEL.BACKBONE.NAME, pretrained=True, aux_logits=False)
        net = CBM(backbone=backbone, num_concepts=num_attrs, num_classes=num_classes)
        state_dict = torch.load(cfg.MODEL.CKPT_PATH, map_location=device)
        net.load_state_dict(state_dict)
    elif "backbone" in experiment_name:
        net = Backbone(name=cfg.MODEL.NAME, num_classes=num_classes)
        state_dict = torch.load(cfg.MODEL.CKPT_PATH, map_location=device)
        net.load_state_dict(state_dict)
    else:
        raise NotImplementedError

    ###############
    # Evaluations #
    ###############

    net.to(device)
    net.eval()

    # Test Accuracy
    logger.info("Start task accuracy evaluation...")
    test_accuracy(net, compute_corrects, dataloader_test, len(dataloader_test), device, logger)
    if "backbone" in experiment_name:
        logger.info("DONE!")
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
