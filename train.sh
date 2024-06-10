#!/bin/bash

set -x

python train_net.py --config_path configs/dev_train_multistage_CUB.yaml
python train_net.py --config_path configs/dev_train_CUB.yaml

python train_net.py --config_path configs/dev_train_multistage_CUB.yaml --options MODEL.LOSSES.L_C=1e-1
python train_net.py --config_path configs/dev_train_CUB.yaml --options MODEL.LOSSES.L_C=1e-1

python train_net.py --config_path configs/dev_train_multistage_CUB.yaml --options MODEL.LOSSES.L_CPT=1e-3
python train_net.py --config_path configs/dev_train_CUB.yaml --options MODEL.LOSSES.L_CPT=1e-3

python train_net.py --config_path configs/dev_train_multistage_CUB.yaml --options MODEL.USE_ATTENTION=True
python train_net.py --config_path configs/dev_train_CUB.yaml --options MODEL.USE_ATTENTION=True

python train_net.py --config_path configs/dev_train_multistage_CUB.yaml --options MODEL.LOSSES.L_C=1e-1 MODEL.LOSSES.L_CPT=1e-3
python train_net.py --config_path configs/dev_train_CUB.yaml --options MODEL.LOSSES.L_C=1e-1 MODEL.LOSSES.L_CPT=1e-3

python train_net.py --config_path configs/dev_train_multistage_CUB.yaml --options MODEL.LOSSES.L_C=1e-1 MODEL.USE_ATTENTION=True
python train_net.py --config_path configs/dev_train_CUB.yaml --options MODEL.LOSSES.L_C=1e-1 MODEL.USE_ATTENTION=True

python train_net.py --config_path configs/dev_train_multistage_CUB.yaml --options MODEL.LOSSES.L_CPT=1e-3 MODEL.USE_ATTENTION=True
python train_net.py --config_path configs/dev_train_CUB.yaml --options MODEL.LOSSES.L_CPT=1e-3 MODEL.USE_ATTENTION=True

python train_net.py --config_path configs/dev_train_multistage_CUB.yaml --options MODEL.LOSSES.L_C=1e-1 MODEL.LOSSES.L_CPT=1e-3 MODEL.USE_ATTENTION=True
python train_net.py --config_path configs/dev_train_CUB.yaml --options MODEL.LOSSES.L_C=1e-1 MODEL.LOSSES.L_CPT=1e-3 MODEL.USE_ATTENTION=True



python train_net.py --config_path configs/dev_train_multistage_CUB.yaml --options MODEL.NAME="dev_resnet101" MODEL.BACKBONE="resnet101"
python train_net.py --config_path configs/dev_train_CUB.yaml --options MODEL.NAME="dev_resnet101" MODEL.BACKBONE="resnet101"