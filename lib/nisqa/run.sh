#!/usr/bin/env bash

# conda activate espnet_urgent

. /jet/home/clif/workspace/espnet-urgent/tools/activate_python.sh

abs_path=/ocean/projects/cis210027p/wzhangn/espnet_urgent2024/egs2/urgent24/enh1/
# /ocean/projects/cis210027p/wzhangn/espnet_urgent2024/egs2/urgent24/enh1/exp/enh_train_enh_bsrnn_medium_noncausal_raw/enhanced_test/spk1.scp
expdirs=(
    "exp/enh_train_enh_tfgridnet_6layers_raw"
    # "exp/enh_train_enh_bsrnn_large_noncausal_raw"
    # "exp/enh_train_enh_bsrnn_medium_noncausal_raw"
    # "exp/enh_train_enh_bsrnn_medium_2_noncausal_raw"
    # "exp/enh_train_enh_conv_tasnet_raw"
    # "exp/enh_train_enh_conv_tasnet_mapping_raw"
    # "exp/enh_train_enh_conv_tasnet_large_raw"
)
mkdir -p ./exp/diff_test
python run_predict.py \
    --input_scp exp/enh_train_enh_ncsnpp_raw/enhanced_test/spk1.scp \
    --pretrained_model weights/nisqa.tar --num_workers 0 --bs 10 \
    --output_dir exp/diff_test --mode predict_scp

exit 0
#
# test noisy 
# for x in validation test; do
    # echo "===== dump/raw/${x} ====="
    # mkdir -p ./ref/${x}
    # python run_predict.py \
        # --input_scp ${abs_path}/dump/raw/${x}/wav.scp \
        # --pretrained_model weights/nisqa.tar --num_workers 0 --bs 10 \
        # --output_dir ./ref/${x} --mode predict_scp
# done


for exp in "${expdirs[@]}"; do
    for x in validation test; do
        mkdir -p ./${exp}/${x}
        echo "===== ${exp}/enhanced_${x} ====="
        python run_predict.py \
        --input_scp ${abs_path}/${exp}/enhanced_${x}/spk1.scp \
        --pretrained_model weights/nisqa.tar --num_workers 0 --bs 10 \
        --output_dir ./${exp}/${x} --mode predict_scp
    done
done

