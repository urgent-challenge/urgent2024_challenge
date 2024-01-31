# urgent2024_challenge
Official data preparation scripts for the URGENT 2024 Challenge

## Requirements

- `>8` Cores
- At least `1` GPU (recommended for speedup in DNSMOS calculation)
- XXX GB of free disk space
  - Speech
    - DNS5 speech (original 80 GB + resampled 57 GB): 137 GB
    - CommonVoice English speech (original mp3 82 GB + resampled 509 GB): 591 GB
    - LibriTTS (original 44 GB + resampled 7 GB): 51 GB
    - VCTK: 12 GB
    - WSJ (original sph 15GB + converted 31 GB): 46 GB
  - Noise
    - DNS5 noise (original 58 GB + resampled 35 GB): 93 GB
    - WHAM! noise (48 kHz): 76 GB
    - [optional] EPIC-Sounds noise (original video 1.3 TB + original audio 33 GB + resampled 370 GB): 1.7 TB
  - RIR
    - DNS5 RIRs (48 kHz): 6 GB
  - Others
    - default simulated validation data: ~11 GB

With minimum specs, expects the whole process to take YYY hours.

## Instructions

1. Install environmemnt. Python 3.10 and Torch 2.0.1 are recommended.
   With Anaconda, just run

    conda env create -f environment.yaml
    conda activate urgent

2. Download Commonvoice dataset v11 from https://commonvoice.mozilla.org/en/datasets
    a. Select `Common Voice Corpus 11.0`
    b. Enter your email and check the two mandatory boxes
    c. Right-click the `Download Dataset Bundle` button and select "Copy link"
    d. Enter the following commands in your terminal

        URL="<paste-link>"
        wget $URL -O ./datasets_cv11_en/cv-corpus-11.0-2022-09-21-en.tar.gz
        python ./utils/tar_extractor.py -m 5000 \
            -i ./datasets_cv11_en/cv-corpus-11.0-2022-09-21-en.tar.gz \
            -o ./datasets_cv11_en \
            --skip_existing --skip_errors 

3. Run the script

    ./prepare_espnet_data.sh

4. Install eSpeak-NG (used for the phoneme similarity metric computation)
   - Follow the instructions in https://github.com/espeak-ng/espeak-ng/blob/master/docs/guide.md#linux
