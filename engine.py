import os
import sys
import math
import torch
from torch import nn
import pickle as pkl
from tqdm.auto import tqdm
from typing import Iterable
import torch.nn.functional as F
import lightning as L

class Engine(L.LightningModule):
    def __init__(self, clip_model: nn.Module, model: nn.Module,
                 criterion: nn.Module, lr: float):
        super().__init__()
        self.model = model
        self.clip_model = clip_model
        self.criterion = criterion
        self.lr = lr
        # Freeze CLIP
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def training_step(self, batch, batch_idx):
        im_ids, samples, targets = batch
        samples_encoded = self.clip_model.encode_image(samples)
        concept_logits, class_logits = self.model(samples_encoded)
        
        if self.model.retrieved_concepts is None:
            train_loss, xe_loss, mhl_loss = self.criterion(
                outputs=class_logits,
                targets=targets,
                weights=self.model.prototypes,
                target_dis=self.model.all_concepts
            )
            self.log('total_loss', train_loss)
            self.log('cross_entropy_loss', xe_loss)
            self.log('mahalanobis_loss', mhl_loss)
        else:
            train_loss = F.cross_entropy(class_logits, targets)
            self.log('loss', train_loss)
        
        batch_acc = torch.sum(class_logits.argmax(-1) == targets) / targets.size(0)
        self.log('batch_train_acc', batch_acc)
        
        return train_loss
    
    def validation_step(self, batch, batch_idx):
        im_ids, samples, targets = batch

        samples_encoded = self.clip_model.encode_image(samples)
        concept_logits, class_logits = self.model(samples_encoded)
        test_loss = F.cross_entropy(class_logits, targets)

        preds = torch.argmax(class_logits, dim=-1)
        acc = torch.sum(preds == targets) / targets.size(0)

        self.log("test_loss", test_loss)
        self.log("batch_test_acc", acc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer
