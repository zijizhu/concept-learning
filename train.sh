#!/bin/bash

set -x

python train_net.py --name dev_train_CUB --config_path configs/dev_train_CUB.yaml
python eval.py -e logs/CUB_runs/dev_train_CUB
python train_net.py --name dev_train_CUB_custom --config_path configs/dev_train_CUB_custom.yaml
python eval.py -e logs/CUB_runs/dev_train_CUB_custom
python train_net.py --name dev_train_CUB_full --config_path configs/dev_train_CUB_full.yaml
python eval.py -e logs/CUB_runs/dev_train_CUB_full

python train_net.py --name dev_train_CUB_augmentor --config_path configs/dev_train_CUB.yaml --options DATASET.AUGMENTATION=augmentor OPTIM.EPOCHS=30 OPTIM.EARLY_STOP=5 OPTIM.STEP_SIZE=5
python eval.py -e logs/CUB_runs/dev_train_CUB_augmentor
python train_net.py --name dev_train_CUB_custom_augmentor --config_path configs/dev_train_CUB_custom.yaml --options DATASET.AUGMENTATION=augmentor OPTIM.EPOCHS=30 OPTIM.EARLY_STOP=5 OPTIM.STEP_SIZE=5
python eval.py -e logs/CUB_runs/dev_train_CUB_custom_augmentor
python train_net.py --name dev_train_CUB_full_augmentor --config_path configs/dev_train_CUB_full.yaml --options DATASET.AUGMENTATION=augmentor OPTIM.EPOCHS=30 OPTIM.EARLY_STOP=5 OPTIM.STEP_SIZE=5
python eval.py -e logs/CUB_runs/dev_train_CUB_full_augmentor
