#!/bin/bash
# Description: Download the DNSMOS models from the DNS Challenge repository
# Author: Robin Scheibler
# Date: 2024-01-08

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

mkdir -p DNSMOS/DNSMOS

URL=https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS
# Put all the models in a bash array
MODELS=(bak_ovr.onnx model_v8.onnx sig.onnx sig_bak_ovr.onnx)

for model in ${MODELS[@]}; do
    if [ ! -e "DNSMOS/DNSMOS/$model" ]; then
        echo "[DNSMOS ONNX] downloading model ${URL}/${model}"
        wget -c -O DNSMOS/DNSMOS/$model $URL/$model
    else
        echo "[DNSMOS ONNX] 'DNSMOS/DNSMOS/$model' already exists"
    fi
done
