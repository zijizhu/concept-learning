import os
import clip
import torch
import argparse
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from lightning import Trainer, seed_everything

from engine import Engine
from .datasets.cub_dataset import CUBDataset
from models.concept_retrieval import ConceptRetrievalModel, MahalanobisCriterion


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--batch-size', default=4096, type=int)
    parser.add_argument('--stage-one-epochs', default=5000, type=int)
    parser.add_argument('--stage-two-epochs', default=5000, type=int)
    parser.add_argument('--backbone', type=str, choices=['RN50'], default='RN50')
    parser.add_argument('--retrieval_algo', type=str, choices=['greedy', 'hungarian'], default='hungarian')

    parser.add_argument('--no-reg', action='store_true', help='Train stage 1 without regularization')
    parser.add_argument('--num-concepts', default=None, type=int)

    args = parser.parse_args()
    print(args)

    # Set seeds
    seed_everything(args.seed)
    
    # Load dataset
    train_img_dataset = CUBDataset(args.dataset_dir, encoded=True, split='train')
    train_img_dataloader = DataLoader(train_img_dataset, args.batch_size, shuffle=True)
    test_img_dataset = CUBDataset(args.dataset_dir, encoded=True, split='test')
    test_img_dataloader = DataLoader(test_img_dataset, args.batch_size, shuffle=False)

    concepts_encoded = torch.load(os.path.join(args.dataset_dir, 'concepts_encoded.pt')).to(torch.float32)

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
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    engine = Engine(model=model, criterion=criterion)
    
    # Stage 1 training
    print('Stage 1 training:')
    trainer = Trainer(max_epochs=args.epochs,
                      devices=args.devices)
    
    # Stage 2 training, replace and freeze concept layer, redefine loss
    print('Stage 2 training:')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
