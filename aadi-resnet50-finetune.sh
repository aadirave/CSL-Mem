#!/bin/bash

export SLURM_SUBMIT_DIR=/home/arave/Documents/CSL-Mem/
export SLURM_JOB_NAME="resnet 50 finetuning (arave)"

cd $SLURM_SUBMIT_DIR

module load conda
conda activate csl

python main_finetune.py
