# urgent2024_challenge
Official data preparation scripts for the URGENT 2024 Challenge

## Requirements

- >8 Cores
- 1 GPU
- XXX GB of free disk space

With minimum specs, expects the whole process to take YYY hours.

## Instructions

1. Install environmemnt. Python 3.10 and Torch 2.0.1 are recommended.
   With Anaconda, just run
      conda env create -f environment.yaml
      conda activate urgent
2. Run the script
      ./prepare_espnet_data.sh
