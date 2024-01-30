#!/bin/bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

output_dir="./libritts"
mkdir -p "${output_dir}"

#################################
# Download data
#################################
# Refer to https://www.openslr.org/60/
# download in parallel using xargs
echo "Download LibriTTS data from https://www.openslr.org/60/"
urlbase="https://www.openslr.org/resources/60"
echo "train-clean-100 train-clean-360 dev-clean" | tr " " "\n" \
    | xargs -n 1 -P 3 -I{} \
    wget --no-check-certificate --continue "${urlbase}/{}.tar.gz" -O "${output_dir}/{}.tar.gz"
for x in "${output_dir}"/*.tar.gz; do                                                 
    tar xfv "$x" -C "${output_dir}"
done

#################################
# Data preprocessing
#################################
mkdir -p tmp
OMP_NUM_THREADS=1 python utils/estimate_audio_bandwidth.py \
    --audio_dir "${output_dir}/LibriTTS/train-clean-100/" "${output_dir}/LibriTTS/train-clean-360/" \
    --audio_format wav \
    --chunksize 1000 \
    --nj 8 \
    --outfile tmp/libritts_train.json

OMP_NUM_THREADS=1 python utils/estimate_audio_bandwidth.py \
    --audio_dir "${output_dir}/LibriTTS/dev-clean/" \
    --audio_format wav \
    --chunksize 1000 \
    --nj 8 \
    --outfile tmp/libritts_validation.json

OMP_NUM_THREADS=1 python utils/resample_to_estimated_bandwidth.py \
   --bandwidth_data tmp/libritts_train.json \
   --out_scpfile libritts_resampled_train.scp \
   --outdir "${output_dir}/resampled/train" \
   --nj 8 \
   --chunksize 1000

OMP_NUM_THREADS=1 python utils/resample_to_estimated_bandwidth.py \
   --bandwidth_data tmp/libritts_validation.json \
   --out_scpfile libritts_resampled_validation.scp \
   --outdir "${output_dir}/resampled/dev" \
   --nj 8 \
   --chunksize 1000

python utils/get_libritts_transcript.py \
    --audio_scp libritts_resampled_train.scp \
    --audio_dir "${output_dir}/LibriTTS/train-clean-100/" "${output_dir}/LibriTTS/train-clean-360/" \
    --outfile libritts_resampled_train.text \
    --nj 8

python utils/get_libritts_transcript.py \
    --audio_scp libritts_resampled_validation.scp \
    --audio_dir "${output_dir}/LibriTTS/dev-clean/" \
    --outfile libritts_resampled_validation.text \
    --nj 8

awk '{split($1, arr, "_"); print($1" libritts_"arr[1])}' libritts_resampled_train.scp > libritts_resampled_train.utt2spk
awk '{split($1, arr, "_"); print($1" libritts_"arr[1])}' libritts_resampled_validation.scp > libritts_resampled_validation.utt2spk

#--------------------------------
# Output file:
# -------------------------------
# libritts_resampled_train.scp
#    - scp file containing resampled samples for training
# libritts_resampled_train.utt2spk
#    - speaker mapping for filtered training samples
# libritts_resampled_train.text
#    - transcripts for filtered training samples
# libritts_resampled_validation.scp
#    - scp file containing resampled samples for validation
# libritts_resampled_validation.utt2spk
#    - speaker mapping for filtered validation samples
# libritts_resampled_validation.text
#    - transcripts for filtered validation samples
