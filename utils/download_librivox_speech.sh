#!/usr/bin/bash
# Automatically download the read_speech part of the DNSv5 challenge

filename=read_speech.tgz.parta
subdir=Track1_Headset
output_dir=${1:-dns5_fullband}/${subdir}
NPROC=${2:-8}

mkdir -p ${output_dir}

echo ${output_dir} ${NPROC}

###############################################################

download_dataset () {
    URL=$1
    BASENAME=`basename $URL`
    LOG=${OUTDIR}/${BASENAME}.log
    echo ${OUTDIR} $URL
    wget -ct 0 --retry-connrefused $URL -O ${OUTDIR}/${BASENAME} > $LOG 2>&1
}

# number of parallel processes to use
NPROC=13

export OUTDIR=${output_dir}
export -f download_dataset

mkdir -p ${output_dir}

url="https://dnschallengepublic.blob.core.windows.net/dns5archive/V5_training_dataset/${subdir}"
echo ${url}/${filename}{a..u} | xargs -n 1 -P ${NPROC} bash -c 'download_dataset "$0"'

# concatenate the files
echo "Extract DNSv5 challenge speech data"
cat "${output_dir}"/read_speech.tgz.parta? | python ./utils/tar_extractor.py -m 5000 --pipe -o ${output_dir} --skip_existing --skip_errors
