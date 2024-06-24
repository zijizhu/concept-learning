#! /bin/bash

set -x

mkdir -p datasets/CUB && cd datasets || exit 1

wget -O CUB_200_2011.tgz -q --show-progress "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
tar -zxf CUB_200_2011.tgz -C CUB
mv CUB/attributes.txt CUB/CUB_200_2011/attributes/

cd ../

python data/cub/process_keypoints.py
python data/cub/crop.py
python data/cub/augment.py
