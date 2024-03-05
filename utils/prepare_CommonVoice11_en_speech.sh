#!/bin/bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

dnsmos_model_dir="./DNSMOS"
output_dir="./datasets_cv11_en/cv-corpus-11.0-2022-09-21/en"

echo "=== Preparing CommonVoice data ==="
if [ ! -d "${dnsmos_model_dir}/DNSMOS" ]; then
    echo "Please manually download all models (*.onnx) from https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS/DNSMOS and set the variable 'dnsmos_model_dir'"
    exit 1
fi
#################################
# Download data
#################################
if [ ! -d "${output_dir}/clips" ]; then
    echo "Please manually download the data (Common Voice Corpus 11.0) from https://commonvoice.mozilla.org/en/datasets and save them under the directory '$output_dir'"
    echo "Refer to the README for more details"
    exit 1
fi

#################################
# Data preprocessing
#################################
mkdir -p tmp
BW_EST_FILE=tmp/commonvoice_11.0_en.json
if [ ! -f ${BW_EST_FILE} ]; then
    echo "[CommonVoice] estimating audio bandwidth"
    OMP_NUM_THREADS=1 python utils/estimate_audio_bandwidth.py \
        --audio_dir "${output_dir}/clips/" \
        --audio_format mp3 \
        --chunksize 1000 \
        --nj 16 \
        --outfile ${BW_EST_FILE}
else
    echo "Estimated bandwidth file already exists. Delete ${BW_EST_FILE} if you want to re-estimate."
fi

RESAMP_SCP_FILE=tmp/commonvoice_11.0_en_resampled.scp
if [ ! -f ${RESAMP_SCP_FILE} ]; then
    echo "[CommonVoice] resampling to estimated audio bandwidth"
    OMP_NUM_THREADS=1 python utils/resample_to_estimated_bandwidth.py \
       --bandwidth_data ${BW_EST_FILE} \
       --out_scpfile ${RESAMP_SCP_FILE} \
       --outdir "${output_dir}/resampled" \
       --max_files 5000 \
       --nj 8 \
       --chunksize 1000
else
    echo "Resampled scp file already exists. Delete ${RESAMP_SCP_FILE} if you want to re-resample."
fi

#################################
# Data filtering based on DNSMOS
#################################
DNSMOS_JSON_FILE=tmp/commonvoice_11.0_en_resampled_dnsmos.json
DNSMOS_GZ_FILE="data/`basename ${DNSMOS_JSON_FILE}`.gz"
if [ -f ${DNSMOS_GZ_FILE} ]; then
    gunzip -c ${DNSMOS_GZ_FILE} > ${DNSMOS_JSON_FILE}
fi
if [ ! -f ${DNSMOS_JSON_FILE} ]; then
    echo "[CommonVoice] calculating DNSMOS scores"
    python utils/get_dnsmos.py \
        --json_path "${RESAMP_SCP_FILE}" \
        --outfile "${DNSMOS_JSON_FILE}" \
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
VAD_SCP_FILE=tmp/commonvoice_11.0_en_resampled_filtered_vad.scp
if [ ! -f ${VAD_SCP_FILE} ]; then
    echo "[CommonVoice] filtering via VAD"
    OMP_NUM_THREADS=1 python utils/filter_via_vad.py \
        --scp_path "${RESAMP_SCP_FILE}" \
        --outfile "${VAD_SCP_FILE}" \
        --vad_mode 2 \
        --threshold 0.2 \
        --nj 8 \
        --chunksize 200
else
    echo "VAD scp file already exists. Delete ${VAD_SCP_FILE} if you want to re-estimate."
fi

# remove low-quality samples
FILTERED_SCP_FILE=tmp/commonvoice_11.0_en_resampled_filtered_dnsmos.scp
if [ ! -f ${FILTERED_SCP_FILE} ]; then
    echo "[CommonVoice] filtering via DNSMOS"
    python utils/filter_via_dnsmos.py \
        --scp_path "${VAD_SCP_FILE}" \
        --json_path "${DNSMOS_JSON_FILE}" \
        --outfile "${FILTERED_SCP_FILE}" \
        --score_name OVRL --threshold 3.0 \
        --score_name SIG --threshold 3.0 \
        --score_name BAK --threshold 3.0
else
    echo "Filtered scp file already exists. Delete ${FILTERED_SCP_FILE} if you want to re-estimate."
fi

echo "[CommonVoice] preparing data files"
python utils/get_commonvoice_subset_split.py \
    --scp_path tmp/commonvoice_11.0_en_resampled_filtered_dnsmos.scp \
    --tsv_path "${output_dir}/train.tsv" \
    --outfile commonvoice_11.0_en_resampled_filtered_train.scp

python utils/get_commonvoice_subset_split.py \
    --scp_path tmp/commonvoice_11.0_en_resampled_filtered_dnsmos.scp \
    --tsv_path "${output_dir}/dev.tsv" \
    --outfile commonvoice_11.0_en_resampled_filtered_validation.scp

awk 'FNR==NR {arr[$2]=$1; next} {print($1" cv11_"arr[$1".mp3"])}' \
    "${output_dir}"/train.tsv \
    commonvoice_11.0_en_resampled_filtered_train.scp \
    > commonvoice_11.0_en_resampled_filtered_train.utt2spk
awk 'FNR==NR {arr[$2]=$1; next} {print($1" cv11_"arr[$1".mp3"])}' \
    "${output_dir}"/dev.tsv \
    commonvoice_11.0_en_resampled_filtered_validation.scp \
    > commonvoice_11.0_en_resampled_filtered_validation.utt2spk

python utils/get_commonvoice_transcript.py \
    --audio_scp commonvoice_11.0_en_resampled_filtered_train.scp \
    --tsv_path "${output_dir}/train.tsv" \
    --outfile commonvoice_11.0_en_resampled_filtered_train.text

python utils/get_commonvoice_transcript.py \
    --audio_scp commonvoice_11.0_en_resampled_filtered_validation.scp \
    --tsv_path "${output_dir}/dev.tsv" \
    --outfile commonvoice_11.0_en_resampled_filtered_validation.text

#--------------------------------
# Output file:
# -------------------------------
# commonvoice_11.0_en_resampled_filtered_train.scp
#    - scp file containing filtered samples (after resampling) for training
# commonvoice_11.0_en_resampled_filtered_train.utt2spk
#    - speaker mapping for filtered training samples
# commonvoice_11.0_en_resampled_filtered_train.text
#    - transcript for filtered training samples
# commonvoice_11.0_en_resampled_filtered_validation.scp
#    - scp file containing filtered samples (after resampling) for validation
# commonvoice_11.0_en_resampled_filtered_validation.utt2spk
#    - speaker mapping for filtered validation samples
# commonvoice_11.0_en_resampled_filtered_validation.text
#    - transcript for filtered validation samples
