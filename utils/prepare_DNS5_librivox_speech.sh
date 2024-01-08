#!/bin/bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

dnsmos_model_dir="./DNSMOS"
output_dir="./dns5_fullband"


if [ ! -d "${dnsmos_model_dir}/DNSMOS" ]; then
    echo "Please manually download all models (*.onnx) from https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS/DNSMOS and set the variable 'dnsmos_model_dir'"
    exit 1
fi
#################################
# Download data
#################################
# Refer to https://github.com/microsoft/DNS-Challenge/blob/master/download-dns-challenge-5-headset-training.sh
mkdir -p ${output_dir}/Track1_Headset
for suffix in {a..u}; do
    echo "Downloading part ${suffix} of 21"
    blob_name="Track1_Headset/read_speech.tgz.parta${suffix}"
    url="https://dnschallengepublic.blob.core.windows.net/dns5archive/V5_training_dataset/${blob_name}"
    wget --continue "$url" -O "${output_dir}/${blob_name}"
done
cat "${output_dir}"/read_speech.tgz.parta* | tar -xzv -C "${output_dir}"

#################################
# Data preprocessing
#################################
mkdir -p tmp
python utils/estimate_audio_bandwidth.py \
    --audio_dir ${output_dir}/mnt/dnsv5/clean/read_speech/ \
    --audio_format wav \
    --chunksize 1000 \
    --nj 4 \
    --outfile tmp/dns5_clean_read_speech.json

python utils/resample_to_estimated_bandwidth.py \
   --bandwidth_data tmp/dns5_clean_read_speech.json \
   --out_scpfile tmp/dns5_clean_read_speech_resampled.scp \
   --outdir "${output_dir}/resampled/clean/read_speech" \
   --resample_type "kaiser_best" \
   --nj 4 \
   --chunksize 1000

#################################
# Data filtering based on DNSMOS
#################################
python utils/get_dnsmos.py \
    --json_path "tmp/dns5_clean_read_speech_resampled.scp" \
    --outfile "tmp/dns5_clean_read_speech_resampled_dnsmos.json" \
    --use_gpu True \
    --convert_to_torch True \
    --primary_model "${dnsmos_model_dir}/DNSMOS/sig_bak_ovr.onnx" \
    --p808_model "${dnsmos_model_dir}/DNSMOS/model_v8.onnx" \
    --nsplits 1 \
    --job 1

# remove low-quality samples
python utils/filter_via_dnsmos.py \
    --scp_path "tmp/dns5_clean_read_speech_resampled.scp" \
    --json_path "tmp/dns5_clean_read_speech_resampled_dnsmos.json" \
    --outfile "tmp/dns5_clean_read_speech_resampled_filtered.scp" \
    --score_name BAK --threshold 3.0

# remove non-speech samples
python utils/filter_via_vad.py \
    --scp_path "tmp/dns5_clean_read_speech_resampled_filtered.scp" \
    --outfile "tmp/dns5_clean_read_speech_resampled_filtered_vad.scp" \
    --vad_mode 2 \
    --threshold 0.2 \
    --nj 8 \
    --chunksize 200

sort -u tmp/dns5_clean_read_speech_resampled_filtered_vad.scp | \
    awk '{split($1, arr, "_"); if(arr[5]!="reader"){exit 1;} spk=arr[5]"_"arr[6]; print($1" dns5_"spk)}' > tmp/dns5_clean_read_speech_resampled_filtered_vad.utt2spk
utils/utt2spk_to_spk2utt.pl tmp/dns5_clean_read_speech_resampled_filtered_vad.utt2spk > tmp/dns5_clean_read_speech_resampled_filtered_vad.spk2utt
head -n 90 tmp/dns5_clean_read_speech_resampled_filtered_vad.spk2utt > tmp/dns5_clean_read_speech_resampled_filtered_validation.spk2utt
tail -n +91 tmp/dns5_clean_read_speech_resampled_filtered_vad.spk2utt > tmp/dns5_clean_read_speech_resampled_filtered_train.spk2utt
utils/spk2utt_to_utt2spk.pl tmp/dns5_clean_read_speech_resampled_filtered_validation.spk2utt > dns5_clean_read_speech_resampled_filtered_validation.utt2spk
utils/spk2utt_to_utt2spk.pl tmp/dns5_clean_read_speech_resampled_filtered_train.spk2utt > dns5_clean_read_speech_resampled_filtered_train.utt2spk
utils/filter_scp.pl dns5_clean_read_speech_resampled_filtered_validation.utt2spk tmp/dns5_clean_read_speech_resampled_filtered.scp > dns5_clean_read_speech_resampled_filtered_validation.scp
utils/filter_scp.pl dns5_clean_read_speech_resampled_filtered_train.utt2spk tmp/dns5_clean_read_speech_resampled_filtered.scp > dns5_clean_read_speech_resampled_filtered_train.scp

#--------------------------------
# Output file:
# -------------------------------
# dns5_clean_read_speech_resampled_filtered_train.scp
#    - scp file containing filtered samples (after resampling) for training
# dns5_clean_read_speech_resampled_filtered_train.utt2spk
#    - speaker mapping for filtered training samples
# dns5_clean_read_speech_resampled_filtered_validation.scp
#    - scp file containing filtered samples (after resampling) for validation
# dns5_clean_read_speech_resampled_filtered_validation.utt2spk
#    - speaker mapping for filtered validation samples
