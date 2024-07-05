# urgent2024_challenge
Official data preparation scripts for the [URGENT 2024 Challenge](https://urgent-challenge.github.io/urgent2024/)

## Notes

❗️❗️ Please note that the default generated `data/speech_train` subset is only intended for **dynamic mixing (on-the-fly simulation)** in the ESPnet framework. It has the same content in `spk1.scp` (clean reference speech) and `wav.scp` (noisy speech) files to facilitate on-the-fly simulation of different distortions. So this subset must be used in conjunction with a dynamic mixing configuration (an unofficial example using dynamic mixing can be found at [here](https://github.com/Emrys365/espnet/blob/urgent2024/egs2/urgent24/enh1/conf/tuning/train_enh_bsrnn_large_noncausal_dynamic_mixing.yaml#L34-L64)).
* To use a fixed simulation training set (without dynamic mixing), you could follow the [commented lines](https://github.com/urgent-challenge/urgent2024_challenge/blob/main/prepare_espnet_data.sh#L188-L210) in the [`prepare_espnet_data.sh`](https://github.com/urgent-challenge/urgent2024_challenge/blob/main/prepare_espnet_data.sh) script to generate `data/train`.

❗️❗️[**2024-06-22**] we have updated [utils/prepare_WSJ_speech.sh](https://github.com/urgent-challenge/urgent2024_challenge/tree/main/utils/prepare_WSJ_speech.sh) to remove two WSJ training samples (46uc030b and 47hc0418) which are actually non-speech. They may trigger an error when generating the simulation config using [`simulation/generate_data_param.py`](https://github.com/urgent-challenge/urgent2024_challenge/blob/main/simulation/generate_data_param.py).
* If you have already prepared the WSJ data, all you need to do is to remove two lines from the generated `wsj1_train.*` files:
    > ```bash
    > sed -i -e '/46uc030b/d' -e '/47hc0418/d' data/tmp/wsj_train.*
    > ```

<details><summary>Expand to see all past updates before 2024-06-22</summary><div>

❗️❗️[**2024-06-18**] we have updated [conf/simulation_train.yaml](https://github.com/urgent-challenge/urgent2024_challenge/tree/main/conf/simulation_train.yaml) and [conf/simulation_validation.yaml](https://github.com/urgent-challenge/urgent2024_challenge/tree/main/conf/simulation_validation.yaml) to include the WSJ data for simulation. This is to encourage participants who have the WSJ data license to use the data for generating the training and validation data. If you do not have the WSJ data license, you can ignore this update.

❕❕[**2024-06-18**] we have updated two evaluation-related scripts in `evaluation_metrics/` folder: [`calculate_intrusive_se_metrics.py`](https://github.com/urgent-challenge/urgent2024_challenge/blob/main/evaluation_metrics/calculate_intrusive_se_metrics.py) (for handling infinite SDR values) and [`calculate_wer.py`](https://github.com/urgent-challenge/urgent2024_challenge/blob/main/evaluation_metrics/calculate_wer.py) (for compatibility with the latest [Speech2Text interface](https://github.com/espnet/espnet/blob/master/espnet2/bin/s2t_inference.py) in ESPnet). This will not impact the data preparation part, but you will need to update the codebase when evaluating the model performance.

❗️❗️[**2024-06-14**] we have fixed a bug related to the resampling scripts ([#8](https://github.com/urgent-challenge/urgent2024_challenge/pull/9)). If you have already prepared the DNS5 speech dataset using [`utils/prepare_DNS5_librivox_speech.sh`](https://github.com/urgent-challenge/urgent2024_challenge/blob/main/utils/prepare_DNS5_librivox_speech.sh) or [`prepare_espnet_data.sh`](https://github.com/urgent-challenge/urgent2024_challenge/blob/main/prepare_espnet_data.sh), you will need to update the codebase and regenerate the DNS5 speech data using the latest codebase.
* Before rerunning, some itermediate files need to be removed:
    > ```bash
    > rm tmp/dns5_clean_read_speech*
    > rm -r dns5_fullband/Track1_Headset/resampled/
    > rm -r data/tmp/dns5_librivox_train
    > ```

❗️❗️[**2024-06-14**] we have fixed a bug related to the simulation scripts ([#7](https://github.com/urgent-challenge/urgent2024_challenge/pull/7)). If you have already simulated the dataset using [`simulation/simulate_data_from_param.py`](https://github.com/urgent-challenge/urgent2024_challenge/blob/main/simulation/simulate_data_from_param.py) or [`prepare_espnet_data.sh`](https://github.com/urgent-challenge/urgent2024_challenge/blob/main/prepare_espnet_data.sh), you will need to regenerate the audios using the latest codebase. (No need to rerun [`simulation/generate_data_param.py`](https://github.com/urgent-challenge/urgent2024_challenge/blob/main/simulation/generate_data_param.py) if you have already done so.)

</div></details>

## Requirements

- `>8` Cores
- At least `1` GPU (4 or 8 GPUs are recommended for speedup in DNSMOS or other DNN-based metric calculation)
- At least 1.2 TB of free disk space
  - Speech
    - DNS5 speech (original 131 GB + resampled 187 GB): 318 GB
    - CommonVoice English speech (original mp3 82 GB + resampled 509 GB): 591 GB
    - LibriTTS (original 44 GB + resampled 7 GB): 51 GB
    - VCTK: 12 GB
    - WSJ (original sph 24GB + converted 31 GB): 55 GB
  - Noise
    - DNS5 noise (original 58 GB + resampled 35 GB): 93 GB
    - WHAM! noise (48 kHz): 76 GB
  - RIR
    - DNS5 RIRs (48 kHz): 6 GB
  - Others
    - default simulated validation data: ~16 GB

With minimum specs, expects the whole process to take YYY hours.

## Instructions

0. After cloning this repository, run the following command to initialize the submodules:
    ```bash
    git submodule update --init --recursive
    ```

1. Install environmemnt. Python 3.10 and Torch 2.0.1+ are recommended.
   With Anaconda, just run

    ```bash
    conda env create -f environment.yaml
    conda activate urgent
    ```

    > In case of the following error
    > ```
    >   ERROR: Failed building wheel for pypesq
    > ERROR: Could not build wheels for pypesq, which is required to install pyproject.toml-based projects
    > ```
    > you could manually install [`pypesq`](https://github.com/vBaiCai/python-pesq) in advance via: 
    > (make sure you have `numpy` installed before trying this to avoid compilation errors)
    > ```bash
    > python -m pip install https://github.com/vBaiCai/python-pesq/archive/master.zip
    > ```

2. Download Commonvoice dataset v11 from https://commonvoice.mozilla.org/en/datasets

    a. Select `Common Voice Corpus 11.0`

    b. Enter your email and check the two mandatory boxes

    c. Right-click the `Download Dataset Bundle` button and select "Copy link"

    d. Enter the following commands in your terminal

    ```bash
    URL="<paste-link>"
    wget $URL -O ./datasets_cv11_en/cv-corpus-11.0-2022-09-21-en.tar.gz
    python ./utils/tar_extractor.py -m 5000 \
        -i ./datasets_cv11_en/cv-corpus-11.0-2022-09-21-en.tar.gz \
        -o ./datasets_cv11_en \
        --skip_existing --skip_errors
    ``` 

3. Download WSJ0 and WSJ1 datasets from LDC
    > You will need a LDC license to access the data.
    >
    > For URGENT Challenge participants who want to use the data during the challenge period, please contact the organizers for a temporary LDC license.

    a. Download WSJ0 from https://catalog.ldc.upenn.edu/LDC93s6a

    b. Download WSJ1 from https://catalog.ldc.upenn.edu/LDC94S13A

    c. Uncompress and store the downloaded data to the directories `./wsj/wsj0/` and `./wsj/wsj1/`, respectively.

4. Run the script

    ```bash
    ./prepare_espnet_data.sh
    ```

5. Install eSpeak-NG (used for the phoneme similarity metric computation)
   - Follow the instructions in https://github.com/espeak-ng/espeak-ng/blob/master/docs/guide.md#linux

## Optional: Prepare webdataset

The script `./utils/prepare_wds.py` can store the audio files in a collection
of tar files each containing a predefined number of audio files. This is useful
to reduce the number of IO operations during training. Please see the
[documentation](https://github.com/webdataset/webdataset) of `webdataset` for
more information.

```bash
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
