#!/bin/bash

set -x

python train_net.py --config_path configs/dev_train_CUB.yaml
python train_net.py --config_path configs/dev_train_CUB.yaml --options MODEL.LOSSES.L_C=1e-2
python train_net.py --config_path configs/dev_train_CUB.yaml --options MODEL.ACTIVATION=sigmoid
python train_net.py --config_path configs/dev_train_CUB.yaml --options MODEL.OPTIM.LR=1e-4
python train_net.py --config_path configs/dev_train_CUB.yaml --options MODEL.LOSSES.L_C=1e-2 MODEL.ACTIVATION=sigmoid
python train_net.py --config_path configs/dev_train_CUB.yaml --options MODEL.LOSSES.L_C=1e-2 MODEL.OPTIM.LR=1e-4
python train_net.py --config_path configs/dev_train_CUB.yaml --options MODEL.ACTIVATION=sigmoid MODEL.OPTIM.LR=1e-4
python train_net.py --config_path configs/dev_train_CUB.yaml --options MODEL.LOSSES.L_C=1e-2 MODEL.ACTIVATION=sigmoid MODEL.OPTIM.LR=1e-4