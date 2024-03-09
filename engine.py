import os
import sys
import math
import torch
import pickle as pkl
from tqdm.auto import tqdm
from typing import Iterable
import torch.nn.functional as F
import lightning as L

class Engine(L.LightningModule):
    def __init__(self, model, criterion):
        super().__init__()
        self.model = model
        self.criterion = criterion

    def training_step(self, batch, batch_idx):
        samples, targets = batch

        outputs = self.model(samples)
        
        if self.model.retrieved_concepts is None:
            train_loss, xe_loss, mhl_loss = self.criterion(
                outputs=outputs,
                targets=targets,
                weights=self.model.prototypes,
                target_dis=self.model.all_concepts
            )
            self.log('total_loss', train_loss)
            self.log('cross_entropy_loss', xe_loss)
            self.log('mahalanobis_loss', mhl_loss)
        else:
            train_loss = F.cross_entropy(outputs, targets)
            self.log('loss', train_loss)
        
        batch_acc = torch.sum(outputs.argmax(-1) == targets) / targets.size(0)
        self.log('batch_train_acc', batch_acc)
        
        return train_loss
    
    def test_step(self, batch, batch_idx):
        samples, targets = batch

        outputs = self.model(samples)
        test_loss = F.cross_entropy(outputs, targets)

        preds = torch.argmax(outputs, dim=-1)
        acc = torch.sum(preds == targets) / targets.size(0)

        self.log("test_loss", test_loss)
        self.log("batch_test_acc", acc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def train_one_epoch(
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        data_loader: Iterable,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epoch: int):
    model.to(device).train()
    criterion.to(device).train()

    all_outputs, all_targets = [], []

    for _, samples, targets in data_loader:
        samples, targets = samples.to(device), targets.to(device)

        outputs = model(samples)
        
        if model.retrieved_concepts is None:
            total_loss, xe_loss, mhl_loss = criterion(outputs=outputs,
                                                    targets=targets,
                                                    weights=model.prototypes,
                                                    target_dis=model.all_concepts)
        else:
            loss = criterion(outputs, targets)

        batch_acc = torch.sum(outputs.argmax(-1) == targets) / targets.size(0)

        all_outputs.append(outputs.argmax(-1))
        all_targets.append(targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # gather the stats from all processes
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.inference_mode()
def evaluate(
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        data_loader: Iterable,
        device: torch.device):
    model.eval()
    criterion.eval()

    all_preds, all_tgts = [], []
    header = 'Test: '
    metric_logger = utils.MetricLogger(delimiter="\t")
    for _, samples, targets in metric_logger.log_every(data_loader, 100, header):
        samples, targets = samples.to(device), targets.to(device)
        outputs = model(samples)

        preds = torch.argmax(outputs, dim=-1)
        acc = torch.sum(preds == targets) / targets.size(0)

        all_preds.append(preds)
        all_tgts.append(targets)

        metric_logger.update(test_acc=acc)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    all_preds, all_tgts = torch.cat(all_preds), torch.cat(all_tgts)
    epoch_acc = (torch.sum(all_preds == all_tgts) / len(all_preds) * 100)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, epoch_acc
