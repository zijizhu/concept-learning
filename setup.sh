#!/bin/bash

set -x

apt-get update -y && apt-get upgrade -y

pip install --upgrade ftfy regex tqdm lightning pandas scipy
pip install git+https://github.com/openai/CLIP.git

# Setup datasets
mkdir data && cd data && mkdir CUB_200_2011
wget -q --show-progress "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
tar -xzf CUB_200_2011.tgz -C CUB_200_2011 --no-same-owner