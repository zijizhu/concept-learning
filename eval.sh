#!/bin/sh

for dir in logs/CUB_runs/*/
do
  python eval.py --experiment_dir "${dir}";
done