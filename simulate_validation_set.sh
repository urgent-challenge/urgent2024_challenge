#!/bin/bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

python simulation/generate_data_param.py --config conf/simulation_validation.yaml

python simulation/simulate_data_from_param.py \
    --config simulation/simulation_validation.yaml \
    --meta_tsv simulation_validation/log/meta.tsv \
    --nj 8 \
    --chunksize 1000
