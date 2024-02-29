#!/bin/bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

output_dir="./vctk"
mkdir -p "${output_dir}"

echo "=== Preparing VCTK data ==="
#################################
# Download data
#################################
# Refer to https://datashare.ed.ac.uk/handle/10283/3443
echo "[VCTK] downloading data"
wget --continue "https://datashare.ed.ac.uk/download/DS_10283_3443.zip" -O "${output_dir}/DS_10283_3443.zip"
if [ ! -f "${output_dir}/VCTK-Corpus-0.92.zip" ]; then
    echo "Unzip DS_10283_3443.zip file"
    UNZIP_DISABLE_ZIPBOMB_DETECTION=1 unzip "${output_dir}/DS_10283_3443.zip" -d "${output_dir}"
else
    echo "Skip unzipping DS_10283_3443.zip file"
fi
if [ ! -d "${output_dir}/VCTK-Corpus" ]; then
    echo "Unzip VCTK-Corpus-0.92.zip file"
    UNZIP_DISABLE_ZIPBOMB_DETECTION=1 unzip "${output_dir}/VCTK-Corpus-0.92.zip" -d "${output_dir}/VCTK-Corpus"
else
    echo "Skip unzipping VCTK-Corpus-0.92.zip file"
fi

echo "[VCTK] preparing data files"
for x in p225 p227 p228 p229 p230 p231 p233 p234 p236 p237 p238 p239 p240 p241 p243 p244 p245 p246 p247 p248 p249 p250 p251 p252 p253 p254 p255 p256 p258 p259 p260 p261 p262 p263 p264 p265 p266 p267 p268 p269 p270 p271 p272 p273 p274 p275 p276 p277 p278 p279 p280 p281 p282 p283 p284 p285 p286 p288 p292 p293 p294 p295 p297 p298 p299 p300 p301 p302 p303 p304 p305 p306 p307 p308 p310 p311 p312 p313 p314 p316 p317 p318 p323 p326 p329 p330 p333 p334 p335 p336 p339 p340 p341 p343 p345 p347 p351 p360 p361 p362 p363 p364 p374 p376; do
    find "${output_dir}"/VCTK-Corpus/wav48_silence_trimmed/$x -iname '*.flac'
done | awk -F '/' '{print($NF" 48000 "$0)}' | sed -e 's/\.flac / /g' | sort -u > vctk_train.scp

for x in p226 p287 p315; do
    find "${output_dir}"/VCTK-Corpus/wav48_silence_trimmed/$x -iname '*.flac'
done | awk -F '/' '{print($NF" 48000 "$0)}' | sed -e 's/\.flac / /g' | sort -u > vctk_validation.scp

awk '{split($1, arr, "_"); print($1" vctk_"arr[1])}' vctk_train.scp > vctk_train.utt2spk
awk '{split($1, arr, "_"); print($1" vctk_"arr[1])}' vctk_validation.scp > vctk_validation.utt2spk

python utils/get_vctk_transcript.py \
    --audio_scp vctk_train.scp \
    --vctk_dir "${output_dir}/VCTK-Corpus" \
    --outfile vctk_train.text \
    --nj 8

python utils/get_vctk_transcript.py \
    --audio_scp vctk_validation.scp \
    --vctk_dir "${output_dir}/VCTK-Corpus" \
    --outfile vctk_validation.text \
    --nj 8

#--------------------------------
# Output file:
# -------------------------------
# vctk_train.scp
#    - scp file containing samples for training
# vctk_train.utt2spk
#    - speaker mapping for filtered training samples
# vctk_train.text
#    - transcript for filtered training samples
# vctk_validation.scp
#    - scp file containing samples for validation
# vctk_validation.utt2spk
#    - speaker mapping for filtered validation samples
# vctk_validation.text
#    - transcript for filtered validation samples
