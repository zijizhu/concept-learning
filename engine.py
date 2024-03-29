import os
import sys
import math
import torch
import pickle as pkl
from torch import nn
import lightning as L
from tqdm.auto import tqdm
from typing import Iterable
import torch.nn.functional as F

from models.concept_retrieval import ConceptRetrievalModel

class Engine(L.LightningModule):
    def __init__(self, clip_model: nn.Module, model: ConceptRetrievalModel,
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
        samples_encoded = self.clip_model.encode_image(samples).to(torch.float32)
        concept_logits, class_logits = self.model(samples_encoded)
        
        if self.model.retrieved_concepts is None:
            train_loss, xe_loss, mmd_loss = self.criterion(
                preds=class_logits,
                tgts=targets,
                weights=self.model.prototypes,
                weights_tgt=self.model.all_concepts
            )
            self.log('train_loss', train_loss, on_step=True)
            self.log('cross_entropy_loss', xe_loss, on_step=True)
            self.log('mmd_loss', mmd_loss, on_step=True)
        else:
            train_loss = F.cross_entropy(class_logits, targets)
            self.log('train_loss', train_loss, on_step=True)
        
        batch_acc = torch.sum(class_logits.argmax(-1) == targets) / targets.size(0)
        self.log('batch_train_acc', batch_acc, on_step=True)
        
        return train_loss
    
    def validation_step(self, batch, batch_idx):
        im_ids, samples, targets = batch

        samples_encoded = self.clip_model.encode_image(samples).to(torch.float32)
        concept_logits, class_logits = self.model(samples_encoded)
        test_loss = F.cross_entropy(class_logits, targets)

        preds = torch.argmax(class_logits, dim=-1)
        acc = torch.sum(preds == targets) / targets.size(0)

        self.log("test_loss", test_loss, on_step=True)
        self.log("batch_test_acc", acc, on_step=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer
