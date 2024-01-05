#!/bin/bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

output_dir="./data"

################################
# Speech data
################################
mkdir -p "${output_dir}/tmp"

./utils/prepare_DNS5_librivox_speech.sh
for subset in train; do
    mkdir -p "${output_dir}/tmp/dns5_librivox_${subset}"
    awk '{print $1" "$3}' dns5_clean_read_speech_resampled_filtered_${subset}.scp > "${output_dir}"/tmp/dns5_librivox_${subset}/wav.scp
    cp dns5_clean_read_speech_resampled_filtered_${subset}.utt2spk "${output_dir}"/tmp/dns5_librivox_${subset}/utt2spk
    utils/utt2spk_to_spk2utt.pl "${output_dir}"/tmp/dns5_librivox_${subset}/utt2spk > "${output_dir}"/tmp/dns5_librivox_${subset}/spk2utt
    awk '{print $1" "$2}' dns5_clean_read_speech_resampled_filtered_${subset}.scp > "${output_dir}"/tmp/dns5_librivox_${subset}/utt2fs
    awk '{print $1" 1ch_"$2"Hz"}' dns5_clean_read_speech_resampled_filtered_${subset}.scp > "${output_dir}"/tmp/dns5_librivox_${subset}/utt2category
    cp "${output_dir}"/tmp/dns5_librivox_${subset}/wav.scp "${output_dir}"/tmp/dns5_librivox_${subset}/spk1.scp
done

./utils/prepare_LibriTTS_speech.sh
for subset in train; do
    mkdir -p "${output_dir}/tmp/libritts_${subset}"
    awk '{print $1" "$3}' libritts_resampled_${subset}.scp > "${output_dir}"/tmp/libritts_${subset}/wav.scp
    cp libritts_resampled_${subset}.utt2spk "${output_dir}"/tmp/libritts_${subset}/utt2spk
    utils/utt2spk_to_spk2utt.pl "${output_dir}"/tmp/libritts_${subset}/utt2spk > "${output_dir}"/tmp/libritts_${subset}/spk2utt
    awk '{print $1" "$2}' libritts_resampled_${subset}.scp > "${output_dir}"/tmp/libritts_${subset}/utt2fs
    awk '{print $1" 1ch_"$2"Hz"}' libritts_resampled_${subset}.scp > "${output_dir}"/tmp/libritts_${subset}/utt2category
    cp "${output_dir}"/tmp/libritts_${subset}/wav.scp "${output_dir}"/tmp/libritts_${subset}/spk1.scp
done

./utils/prepare_VCTK_speech.sh
for subset in train; do
    mkdir -p "${output_dir}/tmp/vctk_${subset}"
    awk '{print $1" "$3}' vctk_${subset}.scp > "${output_dir}"/tmp/vctk_${subset}/wav.scp
    cp vctk_${subset}.utt2spk "${output_dir}"/tmp/vctk_${subset}/utt2spk
    utils/utt2spk_to_spk2utt.pl "${output_dir}"/tmp/vctk_${subset}/utt2spk > "${output_dir}"/tmp/vctk_${subset}/spk2utt
    awk '{print $1" "$2}' vctk_${subset}.scp > "${output_dir}"/tmp/vctk_${subset}/utt2fs
    awk '{print $1" 1ch_"$2"Hz"}' vctk_${subset}.scp > "${output_dir}"/tmp/vctk_${subset}/utt2category
    cp "${output_dir}"/tmp/vctk_${subset}/wav.scp "${output_dir}"/tmp/vctk_${subset}/spk1.scp
done

#./utils/prepare_WSJ_speech.sh
#for subset in train; do
#    mkdir -p "${output_dir}/tmp/wsj_${subset}"
#    awk '{print $1" "$3}' wsj_${subset}.scp > "${output_dir}"/tmp/wsj_${subset}/wav.scp
#    cp wsj_${subset}.utt2spk "${output_dir}"/tmp/wsj_${subset}/utt2spk
#    utils/utt2spk_to_spk2utt.pl "${output_dir}"/tmp/wsj_${subset}/utt2spk > "${output_dir}"/tmp/wsj_${subset}/spk2utt
#    awk '{print $1" "$2}' wsj_${subset}.scp > "${output_dir}"/tmp/wsj_${subset}/utt2fs
#    awk '{print $1" 1ch_"$2"Hz"}' wsj_${subset}.scp > "${output_dir}"/tmp/wsj_${subset}/utt2category
#    cp "${output_dir}"/tmp/wsj_${subset}/wav.scp "${output_dir}"/tmp/wsj_${subset}/spk1.scp
#done

# Combine all data
mkdir -p "${output_dir}/speech_train"
utils/combine_data.sh --extra_files "utt2category utt2fs spk1.scp" data/speech_train data/dns5_librivox_train data/libritts_train data/vctk_train #data/wsj_train

################################
# Noise and RIR data
################################
./utils/prepare_DNS5_noise_rir.sh

./utils/prepare_wham_noise.sh

./utils/prepare_epic_sounds_noise.sh

# Combine all data for the training set
awk '{print $3}' data/dns5_noise_resampled_train/wav.scp data/wham_noise_train/wav.scp data/epic_sounds_noise_resampled_train.scp > "${output_dir}/noise_train.scp"
awk '{print $3}' data/dns5_rirs.scp > "${output_dir}/rir_train.scp"

# Data simulation for the validation set
python simulation/generate_data_param.py --config conf/simulation_validation.yaml
# ~30 minutes with nj=8
python simulation/simulate_data_from_param.py \
    --config simulation/simulation_validation.yaml \
    --meta_tsv simulation_validation/log/meta.tsv \
    --nj 8 \
    --chunksize 200

mkdir -p "${output_dir}"/validation
awk -F"\t" 'NR>1{print($1" "$2)}' simulation_validation/log/meta.tsv | sort -u > "${output_dir}"/validation/wav.scp 
awk -F"\t" 'NR>1{print($1" "$4)}' simulation_validation/log/meta.tsv | sort -u > "${output_dir}"/validation/spk1.scp 
awk -F"\t" 'NR==1{for(i=1; i<=NF; i++) {if($i=="fs") {n=i; break}} next} NR>1{print($1" "$n)}' simulation_validation/log/meta.tsv | sort -u > "${output_dir}"/validation/utt2fs
awk '{print($1" 1ch_"$2"Hz")}' "${output_dir}"/validation/utt2fs > "${output_dir}"/validation/utt2category

#--------------------------------
# Output files:
# -------------------------------
# ${output_dir}/speech_train/
#  |- wav.scp
#  |- spk1.scp
#  |- utt2spk
#  |- spk2utt
#  |- utt2fs
#  \- utt2category
#
# ${output_dir}/validation/
#  |- wav.scp
#  |- spk1.scp
#  |- utt2spk
#  |- spk2utt
#  |- utt2fs
#  \- utt2category
#
# ${output_dir}/noise_train.scp
#
# ${output_dir}/rir_train.scp
