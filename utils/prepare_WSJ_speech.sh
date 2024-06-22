#!/bin/bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

output_dir="./wsj"

echo "=== Preparing WSJ data ==="
#################################
# Download data
#################################
# For URGENT 2024 challenge participants, please refer to https://urgent-challenge.github.io/urgent2024/
# to apply for the temporary LDC license to download the WSJ data.
# NOTE: If you download WSJ data using the aforementioned license, the WSJ data must be deleted once the
#       challenge is completed.
if [ ! -d "${output_dir}/wsj0" ]; then
    echo "Please manually download the data from https://catalog.ldc.upenn.edu/LDC93s6a and save them under the directory '$output_dir/wsj0'"
fi
if [ ! -d "${output_dir}/wsj1" ]; then
    echo "Please manually download the data from https://catalog.ldc.upenn.edu/LDC94S13A and save them under the directory '$output_dir/wsj1'"
fi

#################################
# Download sph2pipe
#################################
if ! command -v sph2pipe; then
    echo "Installing sph2pipe from https://www.ldc.upenn.edu/language-resources/tools/sphere-conversion-tools"
    SPH2PIPE_VERSION=2.5

    if [ ! -e sph2pipe_v${SPH2PIPE_VERSION}.tar.gz ]; then
        wget -nv -T 10 -t 3 -O sph2pipe_v${SPH2PIPE_VERSION}.tar.gz \
            "https://github.com/burrmill/sph2pipe/archive/refs/tags/${SPH2PIPE_VERSION}.tar.gz"
    fi

    if [ ! -e sph2pipe-${SPH2PIPE_VERSION} ]; then
        tar --no-same-owner -xzf sph2pipe_v${SPH2PIPE_VERSION}.tar.gz
        rm -rf sph2pipe 2>/dev/null || true
        ln -s sph2pipe-${SPH2PIPE_VERSION} sph2pipe
    fi

    make -C sph2pipe
    sph2pipe=$PWD/sph2pipe/sph2pipe
else
    sph2pipe=sph2pipe
fi

#################################
# Convert sph formats to wav
#################################
if [ ! -e "tmp/wsj_sph2wav.done" ]; then
    echo "[WSJ] converting sph audios to wav"
    find "${output_dir}/wsj0/" -iname '*.wv1' | while read -r fname; do
        # It takes ~23 minutes to finish audio format conversion in "${output_dir}/wsj0_wav"
        fbasename=$(basename "${fname}" | sed -e 's/\.wv1$//i')
        fdir=$(realpath --relative-to="${output_dir}/wsj0/" $(dirname "${fname}"))
        out="${output_dir}/wsj0_wav/${fdir}/${fbasename}.wav"
        mkdir -p "${output_dir}/wsj0_wav/${fdir}"
        "${sph2pipe}" -f wav "${fname}" > "${out}"
    done

    find "${output_dir}/wsj1/" -iname '*.wv1' | while read -r fname; do
        # It takes ~1 hour to finish audio format conversion in "${output_dir}/wsj1_wav"
        fbasename=$(basename "${fname}" | sed -e 's/\.wv1$//i')
        fdir=$(realpath --relative-to="${output_dir}/wsj1/" $(dirname "${fname}"))
        out="${output_dir}/wsj1_wav/${fdir}/${fbasename}.wav"
        mkdir -p "${output_dir}/wsj1_wav/${fdir}"
        "${sph2pipe}" -f wav "${fname}" > "${out}"
    done
    touch tmp/wsj_sph2wav.done
else
    echo "[WSJ] sph format conversion already finished"
fi


#################################
# Data preprocessing
#################################
echo "[WSJ] preparing data files"
mkdir -p tmp

sed -e 's#:wsj1/# wsj1/#g' -e 's#: /wsj1/# wsj1/#g' -e 's#:wsj0/# wsj0/#g' "${output_dir}/wsj1/13-34.1/wsj1/doc/indices/si_tr_s.ndx" "${output_dir}/wsj0/11-13.1/wsj0/doc/indices/train/tr_s_wv1.ndx" | \
    awk -v out="${output_dir}" '{if(substr($1,1,1)!=";"){n=split($2,a,"/"); split(a[n],b,"."); split($1,c,"_"); str=c[1]"-"c[2]"."c[3]; if(substr($2,length($2)-3,4)!=".wv1"){$2=$2".wav";} print(b[1]" "out"/"a[1]"_wav/"str"/"$2)}}' | \
    sed -e 's#\.wv1$#.wav#g' | grep -v -i 11-2.1/wsj0/si_tr_s/401 | sort -u > tmp/wsj_train_si284.scp
# remove samples with a transcript
sed -i -e '/46uc030b /Id' -e '/47hc0418/Id' tmp/wsj_train_si284.scp

sed -e 's#:wsj1/# wsj1/#g' -e 's#: /wsj1/# wsj1/#g' "${output_dir}/wsj1/13-34.1/wsj1/doc/indices/h1_p0.ndx" | \
    awk -v out="${output_dir}" '{if(substr($1,1,1)!=";"){n=split($2,a,"/"); split(a[n],b,"."); split($1,c,"_"); str=c[1]"-"c[2]"."c[3]; if(substr($2,length($2)-3,4)!=".wv1"){$2=$2".wav";} print(b[1]" "out"/"a[1]"_wav/"str"/"$2)}}' | \
    sed -e 's#\.wv1$#.wav#g' | sort -u > tmp/wsj_test_dev93.scp

python utils/get_wsj_transcript.py \
    --audio_scp tmp/wsj_train_si284.scp tmp/wsj_test_dev93.scp \
    --audio_dir "${output_dir}/wsj0/" "${output_dir}/wsj1/" \
    --chunksize 1000 \
    --nj 8 \
    --outfile tmp/wsj_train_si284.text tmp/wsj_test_dev93.text

awk '{res=substr($0, length($1)+2, length($0)); print($1" 16000 "res)}' tmp/wsj_train_si284.scp > wsj_train.scp
cp tmp/wsj_train_si284.text wsj_train.text
awk '{print($1" wsj_"substr($1,1,3))}' wsj_train.scp > wsj_train.utt2spk

awk '{res=substr($0, length($1)+2, length($0)); print($1" 16000 "res)}' tmp/wsj_test_dev93.scp > wsj_validation.scp
cp tmp/wsj_test_dev93.text wsj_validation.text
awk '{print($1" wsj_"substr($1,1,3))}' wsj_validation.scp > wsj_validation.utt2spk

#--------------------------------
# Output file:
# -------------------------------
# wsj_train.scp
#    - scp file containing samples for training
# wsj_train.utt2spk
#    - speaker mapping for training samples
# wsj_train.text
#    - transcript for training samples
# wsj_validation.scp
#    - scp file containing samples for validation
# wsj_validation.utt2spk
#    - speaker mapping for validation samples
# wsj_validation.text
#    - transcript for validation samples
