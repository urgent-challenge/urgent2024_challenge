#!/bin/bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

output_dir="./libritts"
mkdir -p "${output_dir}"

echo "=== Preparing LibriTTS data ==="
#################################
# Download data
#################################
# Refer to https://www.openslr.org/60/
# download in parallel using xargs
echo "Download LibriTTS data from https://www.openslr.org/60/"
urlbase="https://www.openslr.org/resources/60"
if [ ! -e "${output_dir}/download_libritts.done" ]; then
    echo "train-clean-100 train-clean-360 dev-clean" | tr " " "\n" \
        | xargs -n 1 -P 3 -I{} \
        wget --no-check-certificate --continue "${urlbase}/{}.tar.gz" -O "${output_dir}/{}.tar.gz"
    for x in "${output_dir}"/*.tar.gz; do                                                 
        tar xfv "$x" -C "${output_dir}"
    done
else
    echo "Skip downloading LibriTTS as it has already finished"
fi
touch "${output_dir}"/download_libritts.done

#################################
# Data preprocessing
#################################
mkdir -p tmp
BW_EST_FILE=tmp/libritts_train.json
if [ ! -f ${BW_EST_FILE} ]; then
    echo "[LibriTTS-train] estimating audio bandwidth"
    OMP_NUM_THREADS=1 python utils/estimate_audio_bandwidth.py \
        --audio_dir "${output_dir}/LibriTTS/train-clean-100/" "${output_dir}/LibriTTS/train-clean-360/" \
        --audio_format wav \
        --chunksize 1000 \
        --nj 8 \
        --outfile "${BW_EST_FILE}"
else
    echo "Estimated bandwidth file already exists. Delete ${BW_EST_FILE} if you want to re-estimate."
fi

RESAMP_SCP_FILE=libritts_resampled_train.scp
if [ ! -f ${RESAMP_SCP_FILE} ]; then
    echo "[LibriTTS-train] resampling to estimated audio bandwidth"
    OMP_NUM_THREADS=1 python utils/resample_to_estimated_bandwidth.py \
        --bandwidth_data "${BW_EST_FILE}" \
        --out_scpfile "${RESAMP_SCP_FILE}" \
        --outdir "${output_dir}/resampled/train" \
        --nj 8 \
        --chunksize 1000
else
    echo "Resampled scp file already exists. Delete ${RESAMP_SCP_FILE} if you want to re-resample."
fi

BW_EST_FILE=tmp/libritts_validation.json
if [ ! -f ${BW_EST_FILE} ]; then
    echo "[LibriTTS-validation] estimating audio bandwidth"
    OMP_NUM_THREADS=1 python utils/estimate_audio_bandwidth.py \
        --audio_dir "${output_dir}/LibriTTS/dev-clean/" \
        --audio_format wav \
        --chunksize 1000 \
        --nj 8 \
        --outfile "${BW_EST_FILE}"
else
    echo "Estimated bandwidth file already exists. Delete ${BW_EST_FILE} if you want to re-estimate."
fi

RESAMP_SCP_FILE=libritts_resampled_validation.scp
if [ ! -f ${RESAMP_SCP_FILE} ]; then
    echo "[LibriTTS-train] resampling to estimated audio bandwidth"
    OMP_NUM_THREADS=1 python utils/resample_to_estimated_bandwidth.py \
        --bandwidth_data "${BW_EST_FILE}" \
        --out_scpfile "${RESAMP_SCP_FILE}" \
        --outdir "${output_dir}/resampled/dev" \
        --nj 8 \
        --chunksize 1000
else
    echo "Resampled scp file already exists. Delete ${RESAMP_SCP_FILE} if you want to re-resample."
fi

echo "[LibriTTS] preparing data files"
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
