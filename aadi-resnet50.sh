#!/bin/bash

export SLURM_SUBMIT_DIR=/home/arave/Documents/CSL-Mem/
export SLURM_JOB_NAME="resnet 50 training (arave)"

cd $SLURM_SUBMIT_DIR

module load conda
conda activate csl

python main.py
