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
for subset in train validation; do
    mkdir -p "${output_dir}/tmp/dns5_librivox_${subset}"
    awk '{print $1" "$3}' dns5_clean_read_speech_resampled_filtered_${subset}.scp > "${output_dir}"/tmp/dns5_librivox_${subset}/wav.scp
    cp dns5_clean_read_speech_resampled_filtered_${subset}.utt2spk "${output_dir}"/tmp/dns5_librivox_${subset}/utt2spk
    utils/utt2spk_to_spk2utt.pl "${output_dir}"/tmp/dns5_librivox_${subset}/utt2spk > "${output_dir}"/tmp/dns5_librivox_${subset}/spk2utt
    awk '{print $1" "$2}' dns5_clean_read_speech_resampled_filtered_${subset}.scp > "${output_dir}"/tmp/dns5_librivox_${subset}/utt2fs
    awk '{print $1" 1ch_"$2}' dns5_clean_read_speech_resampled_filtered_${subset}.scp > "${output_dir}"/tmp/dns5_librivox_${subset}/utt2category
    cp "${output_dir}"/tmp/dns5_librivox_${subset}/wav.scp "${output_dir}"/tmp/dns5_librivox_${subset}/spk1.scp
done

./utils/prepare_LibriTTS_speech.sh
for subset in train validation; do
    mkdir -p "${output_dir}/tmp/libritts_${subset}"
    awk '{print $1" "$3}' libritts_resampled_${subset}.scp > "${output_dir}"/tmp/libritts_${subset}/wav.scp
    cp libritts_resampled_${subset}.utt2spk "${output_dir}"/tmp/libritts_${subset}/utt2spk
    utils/utt2spk_to_spk2utt.pl "${output_dir}"/tmp/libritts_${subset}/utt2spk > "${output_dir}"/tmp/libritts_${subset}/spk2utt
    awk '{print $1" "$2}' libritts_resampled_${subset}.scp > "${output_dir}"/tmp/libritts_${subset}/utt2fs
    awk '{print $1" 1ch_"$2}' libritts_resampled_${subset}.scp > "${output_dir}"/tmp/libritts_${subset}/utt2category
    cp "${output_dir}"/tmp/libritts_${subset}/wav.scp "${output_dir}"/tmp/libritts_${subset}/spk1.scp
done

./utils/prepare_VCTK_speech.sh
for subset in train validation; do
    mkdir -p "${output_dir}/tmp/vctk_${subset}"
    awk '{print $1" "$3}' vctk_${subset}.scp > "${output_dir}"/tmp/vctk_${subset}/wav.scp
    cp vctk_${subset}.utt2spk "${output_dir}"/tmp/vctk_${subset}/utt2spk
    utils/utt2spk_to_spk2utt.pl "${output_dir}"/tmp/vctk_${subset}/utt2spk > "${output_dir}"/tmp/vctk_${subset}/spk2utt
    awk '{print $1" "$2}' vctk_${subset}.scp > "${output_dir}"/tmp/vctk_${subset}/utt2fs
    awk '{print $1" 1ch_"$2}' vctk_${subset}.scp > "${output_dir}"/tmp/vctk_${subset}/utt2category
    cp "${output_dir}"/tmp/vctk_${subset}/wav.scp "${output_dir}"/tmp/vctk_${subset}/spk1.scp
done

./utils/prepare_WSJ_speech.sh
for subset in train validation; do
    mkdir -p "${output_dir}/tmp/wsj_${subset}"
    awk '{print $1" "$3}' wsj_${subset}.scp > "${output_dir}"/tmp/wsj_${subset}/wav.scp
    cp wsj_${subset}.utt2spk "${output_dir}"/tmp/wsj_${subset}/utt2spk
    utils/utt2spk_to_spk2utt.pl "${output_dir}"/tmp/wsj_${subset}/utt2spk > "${output_dir}"/tmp/wsj_${subset}/spk2utt
    awk '{print $1" "$2}' wsj_${subset}.scp > "${output_dir}"/tmp/wsj_${subset}/utt2fs
    awk '{print $1" 1ch_"$2}' wsj_${subset}.scp > "${output_dir}"/tmp/wsj_${subset}/utt2category
    cp "${output_dir}"/tmp/wsj_${subset}/wav.scp "${output_dir}"/tmp/wsj_${subset}/spk1.scp
done

# Combine all data
mkdir -p "${output_dir}/speech_train"
utils/combine_data.sh --extra_files "utt2category utt2fs spk1.scp" data/speech_train data/dns5_librivox_train data/libritts_train data/vctk_train data/wsj_train
mkdir -p "${output_dir}/speech_validation"
utils/combine_data.sh --extra_files "utt2category utt2fs spk1.scp" data/speech_validation data/dns5_librivox_validation data/libritts_validation data/vctk_validation data/wsj_validation

################################
# Noise and RIR data
################################
./utils/prepare_DNS5_noise_rir.sh

./utils/prepare_wham_noise.sh

./utils/prepare_epic_sounds_noise.sh

# Combine all data
awk '{print $3}' data/dns5_noise_resampled_train/wav.scp data/wham_noise_train/wav.scp data/epic_sounds_noise_resampled_train.scp > "${output_dir}/noise_train.scp"
awk '{print $3}' data/dns5_noise_resampled_validation/wav.scp data/wham_noise_validation/wav.scp data/epic_sounds_noise_resampled_validation.scp > "${output_dir}/noise_validation.scp"
awk '{print $3}' data/dns5_rirs_train.scp > "${output_dir}/rir_train.scp"
awk '{print $3}' data/dns5_rirs_validation.scp > "${output_dir}/rir_validation.scp"

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
# ${output_dir}/speech_validation/
#  |- wav.scp
#  |- spk1.scp
#  |- utt2spk
#  |- spk2utt
#  |- utt2fs
#  \- utt2category
#
# ${output_dir}/noise_train.scp
#
# ${output_dir}/noise_validation.scp
#
# ${output_dir}/rir_train.scp
#
# ${output_dir}/rir_validation.scp
