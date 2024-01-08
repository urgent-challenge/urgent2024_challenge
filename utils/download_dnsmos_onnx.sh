#!/bin/bash
# Description: Download the DNSMOS models from the DNS Challenge repository
# Author: Robin Scheibler
# Date: 2024-01-08
mkdir -p DNSMOS/DNSMOS

URL=https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS
# Put all the models in a bash array
MODELS=(bak_ovr.onnx model_v8.onnx sig.onnx sig_bak_ovr.onnx)

for model in ${MODELS[@]}; do
    wget -O DNSMOS/$model $URL/$model
done
