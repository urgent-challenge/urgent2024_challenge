# urgent2024_challenge
Official data preparation scripts for the URGENT 2024 Challenge

## Requirements

- `>8` Cores
- At least `1` GPU (recommended for speedup in DNSMOS calculation)
- XXX GB of free disk space
  - Speech
    - DNS5 speech (original + resampled): GB
    - CommonVoice English speech (original + resampled): GB
    - LibriTTS (original + resampled): GB
    - VCTK: GB
    - WSJ: GB
  - Noise
    - DNS5 noise (original + resampled): GB
    - WHAM! noise: GB
    - EPIC-Sounds noise (original + resampled): GB
  - RIR
    - DNS5 RIRs: GB
  - Others
    - default simulated validation data: ~GB

With minimum specs, expects the whole process to take YYY hours.

## Instructions

1. Install environmemnt. Python 3.10 and Torch 2.0.1 are recommended.
   With Anaconda, just run
      conda env create -f environment.yaml
      conda activate urgent
2. Run the script
      ./prepare_espnet_data.sh
