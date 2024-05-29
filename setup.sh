#! /bin/bash

set -x

mkdir -p datasets/CUB && cd datasets/CUB || exit 1
wget -O CUB_200_2011.tgz -q --show-progress "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
tar -zxf CUB_200_2011.tgz