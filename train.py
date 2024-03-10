import os
import clip
import torch
import argparse
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from lightning import Trainer, seed_everything

from engine import Engine
from dataset.cub_dataset import CUBDataset
from models.concept_retrieval import ConceptRetrievalModel, MahalanobisCriterion


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str)
    parser.add_argument('--concepts_path', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--batch_size', default=4096, type=int)
    parser.add_argument('--stage_one_epochs', default=5000, type=int)
    parser.add_argument('--stage_two_epochs', default=5000, type=int)
    parser.add_argument('--backbone', type=str, choices=['RN50'], default='RN50')
    parser.add_argument('--retrieval_algo', type=str, choices=['greedy', 'hungarian'], default='hungarian')

    parser.add_argument('--num-concepts', default=None, type=int)

    args = parser.parse_args()
    print(args)

    # Set seeds
    seed_everything(args.seed)

    # Load clip model
    clip_model, clip_preprocess = clip.load(args.backbone)
    
    # Load dataset
    train_dataset = CUBDataset(args.dataset_dir, split='train', transforms=clip_preprocess)
    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    test_dataset = CUBDataset(args.dataset_dir, split='test', transforms=clip_preprocess)
    test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle=False)

    concepts_encoded = torch.load(args.concepts_path).to(torch.float32)

    # Number of concepts and classes
    if not args.num_concepts:
        args.num_concepts = concepts_encoded.size(0)
    print ("Number of concepts to search for: ", args.num_concepts)
    num_classes = 200

    # Model, loss and optimizer
    model = ConceptRetrievalModel(concepts_encoded,
                                  k=args.num_concepts,
                                  num_classes=num_classes,
                                  retrieval_algo=args.retrieval_algo)
    criterion = MahalanobisCriterion()

    engine = Engine(clip_model=clip_model,
                    model=model,
                    criterion=criterion,
                    lr=args.lr)
    
    # Stage 1 training
    print('Stage 1 training:')
    stage1_trainer = Trainer(max_epochs=args.stage_one_epochs,
                             min_epochs=3000,
                             accelerator=args.device)
    
    stage1_trainer.fit(model=engine,
                       train_dataloaders=train_dataloader,
                       val_dataloaders=test_dataloader)
    
    # Retrieve concepts after stage 1 training
    model.match_concepts()
    
    # Stage 2 training
    print('Stage 2 training:')
    stage2_trainer = Trainer(max_epochs=args.stage_one_epochs,
                             min_epochs=3000,
                             accelerator=args.device)

    stage2_trainer.fit(model=engine,
                       train_dataloaders=train_dataloader,
                       val_dataloader=test_dataloader)
