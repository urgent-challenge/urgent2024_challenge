#!/bin/bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

output_dir="./dns5_fullband"

#################################
# DNS5 noise and RIRs
#################################
# Refer to https://github.com/microsoft/DNS-Challenge/blob/master/download-dns-challenge-5-noise-ir.sh
BLOB_NAMES=(
    noise_fullband/datasets_fullband.noise_fullband.audioset_000.tar.bz2
    noise_fullband/datasets_fullband.noise_fullband.audioset_001.tar.bz2
    noise_fullband/datasets_fullband.noise_fullband.audioset_002.tar.bz2
    noise_fullband/datasets_fullband.noise_fullband.audioset_003.tar.bz2
    noise_fullband/datasets_fullband.noise_fullband.audioset_004.tar.bz2
    noise_fullband/datasets_fullband.noise_fullband.audioset_005.tar.bz2
    noise_fullband/datasets_fullband.noise_fullband.audioset_006.tar.bz2

    noise_fullband/datasets_fullband.noise_fullband.freesound_000.tar.bz2
    noise_fullband/datasets_fullband.noise_fullband.freesound_001.tar.bz2

    datasets_fullband.impulse_responses_000.tar.bz2
)
for blob_name in ${BLOB_NAMES[@]}; do
    url="https://dnschallengepublic.blob.core.windows.net/dns5archive/V5_training_dataset/${blob_name}"
    wget --continue "$url" -O "${output_dir}/${blob_name}"
done
for x in "${output_dir}"/noise_fullband/*.tar.bz2; do                                                 
    tar xfv "$x" -C "${output_dir}"
done
tar xfv "${output_dir}"/datasets_fullband.impulse_responses_000.tar.bz2 -C "${output_dir}"

#################################
# Data preprocessing
#################################
mkdir -p tmp
python utils/estimate_audio_bandwidth.py \
    --audio_dir ${output_dir}/datasets_fullband/noise_fullband/ \
    --audio_format wav \
    --chunksize 1000 \
    --nj 4 \
    --outfile tmp/dns5_noise.json

python utils/resample_to_estimated_bandwidth.py \
   --bandwidth_data tmp/dns5_noise.json \
   --out_scpfile tmp/dns5_noise_resampled.scp \
   --outdir "${output_dir}/resampled/noise" \
   --resample_type "kaiser_best" \
   --nj 4 \
   --chunksize 1000
 
python - <<'EOF'
from collections import defaultdict

data = defaultdict(list)
with open("tmp/dns5_noise_resampled.scp", "r") as f:
    for line in f:
        fs = line.strip().split()[1]
        data[fs].append(line)
lines_cv = []
lines_tr = []
for fs, lst in data.items():
    lst = sorted(lst)
    for line in lst[:len(lst) // 8]:
        lines_cv.append(line)
    for line in lst[len(lst) // 8:]:
        lines_tr.append(line)
with open("dns5_noise_resampled_validation.scp", "w") as f:
    for line in sorted(lines_cv):
        f.write(line)
with open("dns5_noise_resampled_train.scp", "w") as f:
    for line in sorted(lines_tr):
        f.write(line)
EOF

find "${output_dir}/datasets_fullband/impulse_response/" -iname '*.wav' | \
    awk -F'/' '{fname=substr($NF, 1, length($NF)-4); fname=$(NF-2)"-"fname; print(fname" 48000 "$0)}' | \
    sort -u > dns5_rirs.scp

#--------------------------------
# Output file:
# -------------------------------
# dns5_noise_resampled_train.scp
#    - scp file containing resampled noise samples for training
# dns5_noise_resampled_validation.scp
#    - scp file containing resampled noise samples for validation
# dns5_rirs_train.scp
#    - scp file containing RIRs for training
# dns5_rirs_validation.scp
#    - scp file containing RIRs for validation
