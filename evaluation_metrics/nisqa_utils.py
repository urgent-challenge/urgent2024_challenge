#!/usr/bin/env python3
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.append(str(Path(__file__).parent.parent / "lib/NISQA"))
import nisqa.NISQA_lib as NL


def load_nisqa_model(model_path, device="cpu"):
    checkpoint = torch.load(model_path, map_location=torch.device(device))
    args = checkpoint["args"]

    if args["model"] == "NISQA_DIM":
        args["dim"] = True
        args["csv_mos_train"] = None  # column names hardcoded for dim models
        args["csv_mos_val"] = None
    else:
        args["dim"] = False

    if args["model"] == "NISQA_DE":
        args["double_ended"] = True
    else:
        args["double_ended"] = False
        args["csv_ref"] = None

    # Load Model
    model_args = {
        "ms_seg_length": args["ms_seg_length"],
        "ms_n_mels": args["ms_n_mels"],
        "cnn_model": args["cnn_model"],
        "cnn_c_out_1": args["cnn_c_out_1"],
        "cnn_c_out_2": args["cnn_c_out_2"],
        "cnn_c_out_3": args["cnn_c_out_3"],
        "cnn_kernel_size": args["cnn_kernel_size"],
        "cnn_dropout": args["cnn_dropout"],
        "cnn_pool_1": args["cnn_pool_1"],
        "cnn_pool_2": args["cnn_pool_2"],
        "cnn_pool_3": args["cnn_pool_3"],
        "cnn_fc_out_h": args["cnn_fc_out_h"],
        "td": args["td"],
        "td_sa_d_model": args["td_sa_d_model"],
        "td_sa_nhead": args["td_sa_nhead"],
        "td_sa_pos_enc": args["td_sa_pos_enc"],
        "td_sa_num_layers": args["td_sa_num_layers"],
        "td_sa_h": args["td_sa_h"],
        "td_sa_dropout": args["td_sa_dropout"],
        "td_lstm_h": args["td_lstm_h"],
        "td_lstm_num_layers": args["td_lstm_num_layers"],
        "td_lstm_dropout": args["td_lstm_dropout"],
        "td_lstm_bidirectional": args["td_lstm_bidirectional"],
        "td_2": args["td_2"],
        "td_2_sa_d_model": args["td_2_sa_d_model"],
        "td_2_sa_nhead": args["td_2_sa_nhead"],
        "td_2_sa_pos_enc": args["td_2_sa_pos_enc"],
        "td_2_sa_num_layers": args["td_2_sa_num_layers"],
        "td_2_sa_h": args["td_2_sa_h"],
        "td_2_sa_dropout": args["td_2_sa_dropout"],
        "td_2_lstm_h": args["td_2_lstm_h"],
        "td_2_lstm_num_layers": args["td_2_lstm_num_layers"],
        "td_2_lstm_dropout": args["td_2_lstm_dropout"],
        "td_2_lstm_bidirectional": args["td_2_lstm_bidirectional"],
        "pool": args["pool"],
        "pool_att_h": args["pool_att_h"],
        "pool_att_dropout": args["pool_att_dropout"],
    }

    if args["double_ended"]:
        model_args.update(
            {
                "de_align": args["de_align"],
                "de_align_apply": args["de_align_apply"],
                "de_fuse_dim": args["de_fuse_dim"],
                "de_fuse": args["de_fuse"],
            }
        )

    if args["model"] == "NISQA":
        model = NL.NISQA(**model_args)
    elif args["model"] == "NISQA_DIM":
        model = NL.NISQA_DIM(**model_args)
    elif args["model"] == "NISQA_DE":
        model = NL.NISQA_DE(**model_args)
    else:
        raise NotImplementedError("Model not available")

    # Load weights
    missing_keys, unexpected_keys = model.load_state_dict(
        checkpoint["model_state_dict"], strict=True
    )
    if missing_keys:
        print("[NISQA] missing_keys:")
        print(missing_keys)
    if unexpected_keys:
        print("[NISQA] unexpected_keys:")
        print(unexpected_keys)
    model.args = args
    model.device = device
    return model


def predict_nisqa(model, audio_path):
    # ported from https://github.com/gabrielmittag/NISQA/blob/master/nisqa/NISQA_model.py
    data_dir = Path(audio_path).parent
    file_name = Path(audio_path).name
    # The audio will be resampled to 48 kHz
    df_val = pd.DataFrame([file_name], columns=["deg"])
    dataset = NL.SpeechQualityDataset(
        df_val,
        df_con=None,
        data_dir=data_dir,
        filename_column="deg",
        mos_column="predict_only",
        seg_length=model.args["ms_seg_length"],
        max_length=model.args["ms_max_segments"],
        to_memory=None,
        to_memory_workers=None,
        seg_hop_length=model.args["ms_seg_hop_length"],
        transform=None,
        ms_n_fft=model.args["ms_n_fft"],
        ms_hop_length=model.args["ms_hop_length"],
        ms_win_length=model.args["ms_win_length"],
        ms_n_mels=model.args["ms_n_mels"],
        ms_sr=model.args["ms_sr"],
        ms_fmax=model.args["ms_fmax"],
        ms_channel=None,
        double_ended=model.args["double_ended"],
        dim=model.args["dim"],
        filename_column_ref=None,
    )

    if model.args["dim"] == True:
        y_val_hat, y_val = NL.predict_dim(
            model,
            dataset,
            1,
            model.device,
            num_workers=0,
        )
    else:
        y_val_hat, y_val = NL.predict_mos(
            model,
            dataset,
            1,
            model.device,
            num_workers=0,
        )
    return {
        "mos_pred": dataset.df["mos_pred"].squeeze().tolist(),
        "noi_pred": dataset.df["noi_pred"].squeeze().tolist(),
        "dis_pred": dataset.df["dis_pred"].squeeze().tolist(),
        "col_pred": dataset.df["col_pred"].squeeze().tolist(),
        "loud_pred": dataset.df["loud_pred"].squeeze().tolist(),
    }
