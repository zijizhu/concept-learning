#!/bin/bash

set -x

python train_net.py --name dino_base_losses --config_path configs/dino_train_CUB.yaml --options model.loss.l_cpt=1e-3 model.loss.l_dec=1e-2
python eval_dino.py -e logs/CUB_runs/dino_base_losses

python train_net.py --name dino_base --config_path configs/dino_train_CUB.yaml
python eval_dino.py -e logs/CUB_runs/dino_base

python train_net.py --name dino_base_losses_aug --config_path configs/dino_train_CUB.yaml --options model.loss.l_cpt=1e-3 model.loss.l_dec=1e-2 dataset.agumentation="augment"
python eval_dino.py -e logs/CUB_runs/dino_base_losses_aug

python train_net.py --name dino_base_aug --config_path configs/dino_train_CUB.yaml --options dataset.agumentation="augment"
python eval_dino.py -e logs/CUB_runs/dino_base_aug