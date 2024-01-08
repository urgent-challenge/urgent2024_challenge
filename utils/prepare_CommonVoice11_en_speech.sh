#!/bin/bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

dnsmos_model_dir=
output_dir="./datasets_cv11_en"

if [ ! -d "${dnsmos_model_dir}/DNSMOS" ]; then
    echo "Please manually download all models (*.onnx) from https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS/DNSMOS and set the variable 'dnsmos_model_dir'"
    exit 1
fi
#################################
# Download data
#################################
if [ ! -d "${output_dir}/clips" ]; then
    echo "Please manually download the data from https://commonvoice.mozilla.org/en/datasets and save them under the directory '$output_dir'"
    exit 1
fi

#################################
# Data preprocessing
#################################
mkdir -p tmp
python utils/estimate_audio_bandwidth.py \
    --audio_dir "${output_dir}/clips/" \
    --audio_format mp3 \
    --chunksize 1000 \
    --nj 16 \
    --outfile tmp/commonvoice_11.0_en.json

python utils/resample_to_estimated_bandwidth.py \
   --bandwidth_data tmp/commonvoice_11.0_en.json \
   --out_scpfile tmp/commonvoice_11.0_en_resampled.scp \
   --outdir "${output_dir}/resampled" \
   --resample_type "kaiser_best" \
   --nj 8 \
   --chunksize 1000

#################################
# Data filtering based on DNSMOS
#################################
python utils/get_dnsmos.py \
    --json_path "tmp/commonvoice_11.0_en_resampled.scp" \
    --outfile "tmp/commonvoice_11.0_en_resampled_dnsmos.json" \
    --use_gpu True \
    --convert_to_torch True \
    --primary_model "${dnsmos_model_dir}/DNSMOS/sig_bak_ovr.onnx" \
    --p808_model "${dnsmos_model_dir}/DNSMOS/model_v8.onnx" \
    --nsplits 1 \
    --job 1

# remove low-quality samples
python utils/filter_via_dnsmos.py \
    --scp_path "tmp/commonvoice_11.0_en_resampled.scp" \
    --json_path "tmp/commonvoice_11.0_en_resampled_dnsmos.json" \
    --outfile "tmp/commonvoice_11.0_en_resampled_filtered.scp" \
    --score_name OVRL --threshold 3.0 \
    --score_name SIG --threshold 3.0 \
    --score_name BAK --threshold 3.0

# remove non-speech samples
python utils/filter_via_vad.py \
    --scp_path "tmp/commonvoice_11.0_en_resampled_filtered.scp" \
    --outfile "tmp/commonvoice_11.0_en_resampled_filtered_vad.scp" \
    --vad_mode 2 \
    --threshold 0.2 \
    --nj 8 \
    --chunksize 200

python utils/get_commonvoice_subset_split.py \
    --scp_path tmp/commonvoice_11.0_en_resampled_filtered_vad.scp \
    --tsv_path "${output_dir}/train.tsv" \
    --outfile commonvoice_11.0_en_resampled_filtered_train.scp

python utils/get_commonvoice_subset_split.py \
    --scp_path tmp/commonvoice_11.0_en_resampled_filtered_vad.scp \
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

#--------------------------------
# Output file:
# -------------------------------
# commonvoice_11.0_en_resampled_filtered_train.scp
#    - scp file containing filtered samples (after resampling) for training
# commonvoice_11.0_en_resampled_filtered_train.utt2spk
#    - speaker mapping for filtered training samples
# commonvoice_11.0_en_resampled_filtered_validation.scp
#    - scp file containing filtered samples (after resampling) for validation
# commonvoice_11.0_en_resampled_filtered_validation.utt2spk
#    - speaker mapping for filtered validation samples
