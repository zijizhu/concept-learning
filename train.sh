#!/bin/bash

set -x

python train_net.py --config_path configs/dev_train_multistage_CUB.yaml --options MODEL.USE_ATTENTION=True
python train_net.py --config_path configs/dev_train_CUB.yaml --options MODEL.LOSSES.L_C=1e-3
python train_net.py --config_path configs/dev_train_CUB.yaml --options MODEL.ACTIVATION=sigmoid
python train_net.py --config_path configs/dev_train_CUB.yaml --options OPTIM.LR=1e-4
python train_net.py --config_path configs/dev_train_CUB.yaml --options MODEL.LOSSES.L_C=1e-3 MODEL.ACTIVATION=sigmoid
python train_net.py --config_path configs/dev_train_CUB.yaml --options MODEL.LOSSES.L_C=1e-3 OPTIM.LR=1e-4
python train_net.py --config_path configs/dev_train_CUB.yaml --options MODEL.ACTIVATION=sigmoid OPTIM.LR=1e-4
python train_net.py --config_path configs/dev_train_CUB.yaml --options MODEL.LOSSES.L_C=1e-3 MODEL.ACTIVATION=sigmoid OPTIM.LR=1e-4
