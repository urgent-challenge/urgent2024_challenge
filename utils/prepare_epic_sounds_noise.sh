#!/bin/bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

output_dir="./epic_sounds_noise"

#################################
# EPIC-SOUNDS noise (~48 kHz)
#################################
if [ ! -d "${output_dir}/EPIC-KITCHENS-100/P01" ]; then
    # echo "Please manually download the EPIC-KITCHENS-100 data by following https://github.com/epic-kitchens/epic-kitchens-download-scripts and put them under the directory '${output_dir}/EPIC-KITCHENS-100/'"
    git clone https://github.com/epic-kitchens/epic-kitchens-download-scripts "${output_dir}/epic-kitchens-download-scripts"
    python "${output_dir}"/epic-kitchens-download-scripts/epic_downloader.py --train --val --output-path "${output_dir}"
fi

# extract audio from video files with the "original" sampling frequency
if ! command -v ffmpeg; then
    echo "Please manually install 'ffmpeg' to proceed."
    exit 1
fi
find "${output_dir}/EPIC-KITCHENS-100/" -iname '*.mp4' | while read -r fname; do
    fbasename=$(basename "${fname}" | sed -e 's/\.mp4$//i')
    fdir=$(realpath --relative-to="${output_dir}/EPIC-KITCHENS-100/" $(dirname "${fname}"))
    out="${output_dir}/audios/${fdir}/${fbasename}.wav"
    mkdir -p "${output_dir}/audios/${fdir}"
    ffmpeg -nostdin -y -hide_banner -loglevel info -nostats -i "$fname" -vn -acodec pcm_s16le -ac 1 "$out"
done

# download annotations
git clone https://github.com/epic-kitchens/epic-sounds-annotations "${output_dir}/epic-sounds-annotations"

#################################
# Data preprocessing
#################################
mkdir -p tmp

find "${output_dir}"/audios/ -iname '*.wav' | \
    awk -F '/' '{print($NF" "$0)}' | sed -e 's/\.wav / /g' | \
    sort -u > tmp/epic_sounds_noise.scp

python utils/get_epic_sounds_subset_split.py \
    --scp_path tmp/epic_sounds_noise.scp \
    --csv_path "${output_dir}/epic-sounds-annotations/EPIC_Sounds_train.csv" \
    --outfile tmp/epic_sounds_noise_train.json

python utils/get_epic_sounds_subset_split.py \
    --scp_path tmp/epic_sounds_noise.scp \
    --csv_path "${output_dir}/epic-sounds-annotations/EPIC_Sounds_validation.csv" \
    --outfile tmp/epic_sounds_noise_validation.json

python utils/estimate_audio_bandwidth.py \
    --audio_dir "tmp/epic_sounds_noise_train.json" \
    --audio_format wav \
    --chunksize 1000 \
    --nj 4 \
    --outfile tmp/epic_sounds_noise_train.json

python utils/estimate_audio_bandwidth.py \
    --audio_dir "tmp/epic_sounds_noise_validation.json" \
    --audio_format wav \
    --chunksize 1000 \
    --nj 4 \
    --outfile tmp/epic_sounds_noise_validation.json

python utils/resample_to_estimated_bandwidth.py \
   --bandwidth_data tmp/epic_sounds_noise_train.json \
   --out_scpfile epic_sounds_noise_resampled_train.scp \
   --outdir "${output_dir}/audios_resampled" \
   --resample_type "kaiser_best" \
   --nj 4 \
   --chunksize 1000

python utils/resample_to_estimated_bandwidth.py \
   --bandwidth_data tmp/epic_sounds_noise_validation.json \
   --out_scpfile epic_sounds_noise_resampled_validation.scp \
   --outdir "${output_dir}/audios_resampled" \
   --resample_type "kaiser_best" \
   --nj 4 \
   --chunksize 1000

#--------------------------------
# Output file:
# -------------------------------
# epic_sounds_noise_resampled_train.scp
#    - scp file containing resampled EPIC-SOUNDS noise samples for training
# epic_sounds_noise_resampled_validation.scp
#    - scp file containing resampled EPIC-SOUNDS noise samples for validation
