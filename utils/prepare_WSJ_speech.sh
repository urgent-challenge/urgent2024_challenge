#!/bin/bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

output_dir="./wsj"

#################################
# Download data
#################################
# For URGENT 2024 challenge participants, please refer to https://urgent-challenge.github.io/urgent2024/
# to apply for the temporary LDC license to download the WSJ data.
# NOTE: If you download WSJ data using the aforementioned license, the WSJ data must be deleted once the
#       challenge is completed.
if [ ! -d "${output_dir}/wsj0" ]; then
    echo "Please manually download the data from https://catalog.ldc.upenn.edu/LDC93s6a and save them under the directory '$output_dir/wsj0'"
fi
if [ ! -d "${output_dir}/wsj1" ]; then
    echo "Please manually download the data from https://catalog.ldc.upenn.edu/LDC94S13A and save them under the directory '$output_dir/wsj1'"
fi

#################################
# Data preprocessing
#################################
mkdir -p tmp
python utils/estimate_audio_bandwidth.py \
    --audio_dir "${output_dir}/wsj0/" "${output_dir}/wsj1/" \
    --audio_format wav \
    --chunksize 1000 \
    --nj 4 \
    --outfile tmp/wsj_train.json

python utils/resample_to_estimated_bandwidth.py \
   --bandwidth_data tmp/wsj_train.json \
   --out_scpfile wsj_resampled_train.scp \
   --outdir "${output_dir}/resampled/train" \
   --resample_type "kaiser_best" \
   --nj 4 \
   --chunksize 1000

#--------------------------------
# Output file:
# -------------------------------
# wsj_resampled_train.scp
#    - scp file containing resampled samples for training
# wsj_resampled_train.utt2spk
#    - speaker mapping for resampled training samples
# wsj_resampled_train.text
#    - transcript for resampled training samples
# wsj_resampled_validation.scp
#    - scp file containing resampled samples for validation
# wsj_resampled_validation.utt2spk
#    - speaker mapping for resampled validation samples
# wsj_resampled_validation.text
#    - transcript for resampled validation samples
