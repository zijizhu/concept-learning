import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from lightning import seed_everything
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from data.cub.cub_dataset import CUBDataset, get_transforms_part_discovery, get_transforms_resnet101


def test_interventions(model: nn.Module, dataset_test: CUBDataset, num_groups_to_intervene: list[int],
                       rng: np.random.Generator, logger: logging.Logger, writer: SummaryWriter):
    """Given a dataset and concept learning model, test its ability of responding to test-time interventions"""
    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=4)

    for num_groups in num_groups_to_intervene:
        sampled_group_ids = rng.choice(
            np.arange(len(dataset_test.group_names)),
            size=(len(dataset_test), num_groups)
        )
        preds, labels = [], []
        # Inference loop
        for test_example, group_ids_to_intervene in zip(dataloader_test, sampled_group_ids):
            intervention_mask = np.isin(dataset_test.attribute_group_indices, group_ids_to_intervene)
            result = model.inference(test_example, intervention_mask=intervention_mask)

        # Compute accuracy

        # Save rng data
        np.savez(...)


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
        if cfg.DATASET.TRANSFORMS == "resnet101":
            train_transforms, test_transforms = get_transforms_resnet101()
        elif cfg.DATASET.TRANSFORMS == "part_discovery":
            train_transforms, test_transforms = get_transforms_part_discovery()
        else:
            raise NotImplementedError

        dataset_test = CUBDataset(
            os.path.join(cfg.DATASET.ROOT_DIR, "CUB"),
            use_attr=cfg.DATASET.USE_ATTR,
            num_attrs=cfg.DATASET.NUM_ATTRS,
            split="test",
            groups=cfg.DATASET.GROUPS,
            transforms=test_transforms,
        )
    elif cfg.DATASET.NAME == "CELEB":
        raise NotImplementedError
    else:
        raise NotImplementedError

    ###############
    # Load models #
    ###############

    net = ...

    ###############
    # Evaluations #
    ###############

    net.to(device)
    net.eval()
    # TODO Test Intervention
    logger.info("Start intervention evaluation...")
    num_groups_to_intervene = [0, 4, 8, 12, 16, 20, 24, 28]

    # TODO Test Representation

    logger.info("DONE!")


if __name__ == "__main__":
    main()
