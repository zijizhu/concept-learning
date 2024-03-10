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
    def __init__(self, clip_model, model, criterion):
        super().__init__()
        self.model = model
        self.criterion = criterion

    def training_step(self, batch, batch_idx):
        im_ids, samples, targets = batch

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
        im_ids, samples, targets = batch

        outputs = self.model(samples)
        test_loss = F.cross_entropy(outputs, targets)

        preds = torch.argmax(outputs, dim=-1)
        acc = torch.sum(preds == targets) / targets.size(0)

        self.log("test_loss", test_loss)
        self.log("batch_test_acc", acc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
