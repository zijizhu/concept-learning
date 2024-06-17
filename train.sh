#!/bin/bash

set -x

python train_net. --name dev_train_CUB --config_path configs/dev_train_CUB.yaml
python train_net.py --name dev_train_CUB_custom --config_path configs/dev_train_CUB_custom.yaml
python train_net.py --name dev_train_CUB_full --config_path configs/dev_train_CUB_full.yaml

python train_net.py --name dev_train_CUB_augmentor --config_path configs/dev_train_CUB.yaml --options DATASET.AUGMENTATION=augmentor OPTIM.EPOCHS=30 OPTIM.EARLY_STOP=5
python train_net.py --name dev_train_CUB_custom_augmentor --config_path configs/dev_train_CUB_custom.yaml --options DATASET.AUGMENTATION=augmentor OPTIM.EPOCHS=30 OPTIM.EARLY_STOP=5
python train_net.py --name dev_train_CUB_full_augmentor --config_path configs/dev_train_CUB_full.yaml --options DATASET.AUGMENTATION=augmentor OPTIM.EPOCHS=30 OPTIM.EARLY_STOP=5

python train_net.py --name dev_train_CUB_crop --config_path configs/dev_train_CUB.yaml --options DATASET.AUGMENTATION=crop
python train_net.py --name dev_train_CUB_custom_crop --config_path configs/dev_train_CUB_custom.yaml --options DATASET.AUGMENTATION=crop
python train_net.py --name dev_train_CUB_custom_crop --config_path configs/dev_train_CUB_full.yaml --options DATASET.AUGMENTATION=crop
