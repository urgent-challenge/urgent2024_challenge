#!/bin/bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

output_dir="./wham_noise_48k"
mkdir -p "${output_dir}"

echo "=== Preparing WHAM! noise data ==="
#################################
# WHAM! noise (48 kHz, unsegmented)
#################################
echo "[WHAM! noise] downloading from https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/high_res_wham.zip"
wget --continue "https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/high_res_wham.zip" -O "${output_dir}/high_res_wham.zip"
if [ ! -e "${output_dir}/download_high_res_wham.done" ]; then
    unzip "${output_dir}/high_res_wham.zip" -d "${output_dir}"
    touch "${output_dir}/download_high_res_wham.done"
fi

echo "[WHAM! noise] preparing data files"

mkdir -p tmp
find "${output_dir}/high_res_wham/audio/" -iname '*.wav' | \
    awk -F'/' '{fname=substr($NF, 1, length($NF)-4); print(fname" 48000 "$0)}' | \
    sort -u > tmp/wham_noise.scp

python utils/get_wham_subset_split.py \
    --scp_path tmp/wham_noise.scp \
    --csv_path "${output_dir}/high_res_wham/high_res_metadata.csv" \
    --outfile wham_noise_train.scp \
    --subset Train

python utils/get_wham_subset_split.py \
    --scp_path tmp/wham_noise.scp \
    --csv_path "${output_dir}/high_res_wham/high_res_metadata.csv" \
    --outfile wham_noise_validation.scp \
    --subset Valid

#--------------------------------
# Output file:
# -------------------------------
# wham_noise_train.scp
#    - scp file containing WHAM! noise samples for training
# wham_noise_validation.scp
#    - scp file containing WHAM! noise samples for validation
