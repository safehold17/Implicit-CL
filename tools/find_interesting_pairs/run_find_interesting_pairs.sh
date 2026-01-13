#!/bin/bash
# Run find_interesting_pairs.py with the correct conda environment

# Activate the dcd-ctrlsim environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dcd-ctrlsim

export PYTHONPATH=/home/chen/workspace/dcd-ctrlsim:$PYTHONPATH

# Run the script with provided arguments
python tools/find_interesting_pairs/find_interesting_pairs.py "$@"
