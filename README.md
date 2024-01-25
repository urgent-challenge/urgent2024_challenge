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

## Optional: Prepare webdataset

The script `./utils/prepare_wds.py` can store the audio files in a collection
of tar files each containing a predefined number of audio files. This is useful
to reduce the number of IO operations during training. Please see the
[documentation](https://github.com/webdataset/webdataset) of `webdataset` for
more information.

```
OMP_NUM_THREADS=1 python ./utils/prepare_wds.py \
    /path/to/urgent_train_24k_wds \
    --files-per-tar 250 \
    --max-workers 8 \
    --scps data/tmp/commonvoice_11.0_en_resampled_filtered_train.scp \
    data/tmp/dns5_clean_read_speech_resampled_filtered_train.scp \
    data/tmp/vctk_train.scp \
    data/tmp/libritts_resampled_train.scp
```
The script can also resample the whole dataset to a unified sampling frequency
with `--sampling-rate <freq_hz>`. This option will not include samples with
sampling frequency lower than the prescribed frequency.
