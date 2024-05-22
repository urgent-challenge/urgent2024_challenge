#!/bin/bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# please do not add a trailing slash
output_dir="./dns5_fullband"

echo "=== Preparing DNS5 noise and RIR data ==="
#################################
# DNS5 noise and RIRs
#################################
# Refer to https://github.com/microsoft/DNS-Challenge/blob/master/download-dns-challenge-5-noise-ir.sh
echo "[DNS5 noise and RIR] downloading"
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
    #wget --continue "$url" -O "${output_dir}/${blob_name}"
    mkdir -p "${output_dir}/$(dirname $blob_name)"
done
# parallel download
url="https://dnschallengepublic.blob.core.windows.net/dns5archive/V5_training_dataset"
echo noise_fullband/datasets_fullband.noise_fullband.audioset_00{0..6}.tar.bz2 \
    | tr " " "\n" \
    | xargs -n 1 -P 7 -I{} wget --continue "$url/{}" -O "${output_dir}/{}"
echo noise_fullband/datasets_fullband.noise_fullband.freesound_00{0,1}.tar.bz2 \
    | tr " " "\n" \
    | xargs -n 1 -P 2 -I{} wget --continue "$url/{}" -O "${output_dir}/{}"
wget --continue "$url/datasets_fullband.impulse_responses_000.tar.bz2" \
    -O "${output_dir}/datasets_fullband.impulse_responses_000.tar.bz2"

# tar --transform : to transform path with SED regular expression
# tar xfv dns5_fullband/noise_fullband/datasets_fullband.noise_fullband.freesound_000.tar.bz2 \
#   -C dns5_fullband/noise_fullband \
#   --transform 's/datasets_fullband\/noise_fullband/freesound_000/'
trans="s/datasets_fullband\/noise_fullband//"
for sub in audioset freesound; do
    if [ "${sub}" = "audioset" ]; then
        n=6
    else
        n=1
    fi
    for idx in $(seq 0 ${n}); do
        archive="${output_dir}/noise_fullband/datasets_fullband.noise_fullband.${sub}_00${idx}.tar.bz2"
        xdir="${output_dir}/datasets_fullband/noise_fullband/${sub}_00${idx}"
        mkdir -p ${xdir}
        echo "Extracting ${archive}"
        tar xf "${archive}" --transform ${trans} -C "${xdir}"
    done
done
echo "Extracing ${output_dir}/datasets_fullband.impulse_responses_000.tar.bz2"
tar xf "${output_dir}"/datasets_fullband.impulse_responses_000.tar.bz2 -C "${output_dir}"

#################################
# Data preprocessing
#################################
mkdir -p tmp

BW_EST_FILE=tmp/dns5_noise.json
if [ ! -f ${BW_EST_FILE} ]; then
    echo "[DNS5 noise and RIR] estimating audio bandwidth"
    OMP_NUM_THREADS=1 python utils/estimate_audio_bandwidth.py \
        --audio_dir ${output_dir}/datasets_fullband/noise_fullband/ \
        --audio_format wav \
        --chunksize 1000 \
        --nj 8 \
        --outfile "${BW_EST_FILE}"
else
    echo "Estimated bandwidth file already exists. Delete ${BW_EST_FILE} if you want to re-estimate."
fi

RESAMP_SCP_FILE=tmp/dns5_noise_resampled.scp
if [ ! -f ${RESAMP_SCP_FILE} ]; then
    echo "[DNS5 noise and RIR] resampling to estimated audio bandwidth"
    OMP_NUM_THREADS=1 python utils/resample_to_estimated_bandwidth.py \
        --bandwidth_data "${BW_EST_FILE}" \
        --out_scpfile "${RESAMP_SCP_FILE}" \
        --outdir "${output_dir}/resampled/noise" \
        --nj 8 \
        --chunksize 1000
else
    echo "Resampled scp file already exists. Delete ${RESAMP_SCP_FILE} if you want to re-resample."
fi

echo "[DNS5 noise and RIR] preparing data files"

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

find "${output_dir}/datasets_fullband/impulse_responses/" -iname '*.wav' | \
    awk -F'/' '{fname=substr($NF, 1, length($NF)-4); fname=$(NF-2)"-"fname; print(fname" 48000 "$0)}' | \
    sort -u > dns5_rirs.scp

# split of RIRs into train/validation is missing...

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
