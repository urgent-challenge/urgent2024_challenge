#!/bin/bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

output_dir="./libritts"

#################################
# Download data
#################################
# Refer to https://www.openslr.org/60/
for name in train-clean-100 train-clean-360 dev-clean; do
    url="https://www.openslr.org/resources/60/${name}.tar.gz"
    wget --continue "$url" -O "${output_dir}/${name}.tar.gz"
done
for x in "${output_dir}"/*.tar.gz; do                                                 
    tar xfv "$x" -C "${output_dir}"
done

#################################
# Data preprocessing
#################################
mkdir -p tmp
python utils/estimate_audio_bandwidth.py \
    --audio_dir "${output_dir}/LibriTTS/train-clean-100/" "${output_dir}/LibriTTS/train-clean-360/" \
    --audio_format wav \
    --chunksize 1000 \
    --nj 4 \
    --outfile tmp/libritts_train.json

python utils/estimate_audio_bandwidth.py \
    --audio_dir "${output_dir}/LibriTTS/dev-clean/" \
    --audio_format wav \
    --chunksize 1000 \
    --nj 4 \
    --outfile tmp/libritts_validation.json

python utils/resample_to_estimated_bandwidth.py \
   --bandwidth_data tmp/libritts_train.json \
   --out_scpfile libritts_resampled_train.scp \
   --outdir "${output_dir}/resampled/train" \
   --resample_type "kaiser_best" \
   --nj 4 \
   --chunksize 1000

python utils/resample_to_estimated_bandwidth.py \
   --bandwidth_data tmp/libritts_validation.json \
   --out_scpfile libritts_resampled_validation.scp \
   --outdir "${output_dir}/resampled/dev" \
   --resample_type "kaiser_best" \
   --nj 4 \
   --chunksize 1000

awk '{split($1, arr, "_"); print($1" libritts_"arr[1])}' libritts_resampled_train.scp > libritts_resampled_train.utt2spk
awk '{split($1, arr, "_"); print($1" libritts_"arr[1])}' libritts_resampled_validation.scp > libritts_resampled_validation.utt2spk

#--------------------------------
# Output file:
# -------------------------------
# libritts_resampled_train.scp
#    - scp file containing resampled samples for training
# libritts_resampled_train.utt2spk
#    - speaker mapping for filtered training samples
# libritts_resampled_validation.scp
#    - scp file containing resampled samples for validation
# libritts_resampled_validation.utt2spk
#    - speaker mapping for filtered validation samples
