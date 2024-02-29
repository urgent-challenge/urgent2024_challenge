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

echo "[DNS5 LibriVox] Downloading"

url="https://dnschallengepublic.blob.core.windows.net/dns5archive/V5_training_dataset/${subdir}"
count=$(ls "${output_dir}"/read_speech.tgz.parta? | wc -l)
if [ "$count" = "21" ]; then
    echo "Skipping downloading as all 21 ${output_dir}/read_speech.tgz.parta? files already exist"
else
    echo ${url}/${filename}{a..u} | xargs -n 1 -P ${NPROC} bash -c 'download_dataset "$0"'
fi

# concatenate the files
echo "Extract DNSv5 challenge speech data"
if [ ! -e "${output_dir}/download.done" ]; then
    cat "${output_dir}"/read_speech.tgz.parta? | python ./utils/tar_extractor.py -m 5000 --pipe -o ${output_dir} --skip_existing --skip_errors
else
    echo "Skipping extraction as it has already been done"
fi
touch ${output_dir}/download.done
