#!/bin/bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

dnsmos_model_dir="./DNSMOS"
output_dir="./dns5_fullband"

echo "=== Preparing DNS5 LibriVox speech data ==="
if [ ! -d "${dnsmos_model_dir}/DNSMOS" ]; then
    echo "Please manually download all models (*.onnx) from https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS/DNSMOS and set the variable 'dnsmos_model_dir'"
    exit 1
fi
#################################
# Download data
#################################
# Refer to https://github.com/microsoft/DNS-Challenge/blob/master/download-dns-challenge-5-headset-training.sh
./utils/download_librivox_speech.sh ${output_dir} 8

#################################
# Data preprocessing
#################################
mkdir -p tmp
BW_EST_FILE=tmp/dns5_clean_read_speech.json
if [ ! -f ${BW_EST_FILE} ]; then
    echo "[DNS5 LibriVox] estimating audio bandwidth"
    OMP_NUM_THREADS=1 python utils/estimate_audio_bandwidth.py \
        --audio_dir ${output_dir}/Track1_Headset/mnt/dnsv5/clean/read_speech/ \
        --audio_format wav \
        --chunksize 1000 \
        --nj 8 \
        --outfile ${BW_EST_FILE}
else
    echo "Estimated bandwidth file already exists. Delete ${BW_EST_FILE} if you want to re-estimate."
fi

RESAMP_SCP_FILE=tmp/dns5_clean_read_speech_resampled.scp
if [ ! -f ${RESAMP_SCP_FILE} ]; then
    echo "[DNS5 LibriVox] resampling to estimated audio bandwidth"
    OMP_NUM_THREADS=1 python utils/resample_to_estimated_bandwidth.py \
       --bandwidth_data ${BW_EST_FILE} \
       --out_scpfile ${RESAMP_SCP_FILE} \
       --outdir "${output_dir}/Track1_Headset/resampled/clean/read_speech" \
       --max_files 5000 \
       --nj 8 \
       --chunksize 1000
else
    echo "Resampled scp file already exists. Delete ${RESAMP_SCP_FILE} if you want to re-resample."
fi

#########################################
# Data filtering based on VAD and DNSMOS
#########################################
DNSMOS_JSON_FILE="tmp/dns5_clean_read_speech_resampled_dnsmos.json"
DNSMOS_GZ_FILE="data/`basename ${DNSMOS_JSON_FILE}`.gz"
if [ -f ${DNSMOS_GZ_FILE} ]; then
    gunzip -c ${DNSMOS_GZ_FILE} > ${DNSMOS_JSON_FILE}
fi
if [ ! -f ${DNSMOS_JSON_FILE} ]; then
    # It took around 35 hours with a single RTX 2080 Ti GPU
    echo "[DNS5 LibriVox] calculating DNSMOS scores"
    python utils/get_dnsmos.py \
        --json_path ${RESAMP_SCP_FILE} \
        --outfile ${DNSMOS_JSON_FILE} \
        --use_gpu True \
        --convert_to_torch True \
        --primary_model "${dnsmos_model_dir}/DNSMOS/sig_bak_ovr.onnx" \
        --p808_model "${dnsmos_model_dir}/DNSMOS/model_v8.onnx" \
        --nsplits 1 \
        --job 1
else
    echo "DNSMOS json file already exists. Delete ${DNSMOS_JSON_FILE} if you want to re-estimate."
fi

# remove non-speech samples
VAD_SCP_FILE="tmp/dns5_clean_read_speech_resampled_filtered_vad.scp"
if [ ! -f ${VAD_SCP_FILE} ]; then
    echo "[DNS5 LibriVox] filtering via VAD"
    OMP_NUM_THREADS=1 python utils/filter_via_vad.py \
        --scp_path ${RESAMP_SCP_FILE} \
        --outfile ${VAD_SCP_FILE} \
        --vad_mode 2 \
        --threshold 0.2 \
        --nj 8 \
        --chunksize 200
else
    echo "VAD scp file already exists. Delete ${VAD_SCP_FILE} if you want to re-estimate."
fi

# remove low-quality samples
FILTERED_SCP_FILE="tmp/dns5_clean_read_speech_resampled_filtered_dnsmos.scp"
if [ ! -f ${FILTERED_SCP_FILE} ]; then
    echo "[DNS5 LibriVox] filtering via DNSMOS"
    python utils/filter_via_dnsmos.py \
        --scp_path "${VAD_SCP_FILE}" \
        --json_path "${DNSMOS_JSON_FILE}" \
        --outfile ${FILTERED_SCP_FILE} \
        --score_name BAK --threshold 3.0
else
    echo "Filtered scp file already exists. Delete ${FILTERED_SCP_FILE} if you want to re-estimate."
fi

echo "[DNS5 LibriVox] preparing data files"
sort -u tmp/dns5_clean_read_speech_resampled_filtered_dnsmos.scp | \
    awk '{split($1, arr, "_"); if(arr[5]!="reader"){exit 1;} spk=arr[5]"_"arr[6]; print($1" dns5_"spk)}' > tmp/dns5_clean_read_speech_resampled_filtered_dnsmos.utt2spk
utils/utt2spk_to_spk2utt.pl tmp/dns5_clean_read_speech_resampled_filtered_dnsmos.utt2spk > tmp/dns5_clean_read_speech_resampled_filtered_dnsmos.spk2utt
head -n 90 tmp/dns5_clean_read_speech_resampled_filtered_dnsmos.spk2utt > tmp/dns5_clean_read_speech_resampled_filtered_validation.spk2utt
tail -n +91 tmp/dns5_clean_read_speech_resampled_filtered_dnsmos.spk2utt > tmp/dns5_clean_read_speech_resampled_filtered_train.spk2utt
utils/spk2utt_to_utt2spk.pl tmp/dns5_clean_read_speech_resampled_filtered_validation.spk2utt > dns5_clean_read_speech_resampled_filtered_validation.utt2spk
utils/spk2utt_to_utt2spk.pl tmp/dns5_clean_read_speech_resampled_filtered_train.spk2utt > dns5_clean_read_speech_resampled_filtered_train.utt2spk
utils/filter_scp.pl dns5_clean_read_speech_resampled_filtered_validation.utt2spk tmp/dns5_clean_read_speech_resampled.scp > dns5_clean_read_speech_resampled_filtered_validation.scp
utils/filter_scp.pl dns5_clean_read_speech_resampled_filtered_train.utt2spk tmp/dns5_clean_read_speech_resampled.scp > dns5_clean_read_speech_resampled_filtered_train.scp

awk '{print($1" <not-available>")}' dns5_clean_read_speech_resampled_filtered_train.scp > dns5_clean_read_speech_resampled_filtered_train.text
awk '{print($1" <not-available>")}' dns5_clean_read_speech_resampled_filtered_validation.scp > dns5_clean_read_speech_resampled_filtered_validation.text

#--------------------------------
# Output file:
# -------------------------------
# dns5_clean_read_speech_resampled_filtered_train.scp
#    - scp file containing filtered samples (after resampling) for training
# dns5_clean_read_speech_resampled_filtered_train.utt2spk
#    - speaker mapping for filtered training samples
# dns5_clean_read_speech_resampled_filtered_train.text
#    - transcript for filtered training samples
# dns5_clean_read_speech_resampled_filtered_validation.scp
#    - scp file containing filtered samples (after resampling) for validation
# dns5_clean_read_speech_resampled_filtered_validation.utt2spk
#    - speaker mapping for filtered validation samples
# dns5_clean_read_speech_resampled_filtered_validation.text
#    - transcript for filtered validation samples
