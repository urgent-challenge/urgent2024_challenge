from distutils.util import strtobool
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
import torch
from tqdm import tqdm

from espnet2.enh.layers.dnsmos import DNSMOS_local


METRICS = ("DNSMOS_OVRL",)
TARGET_FS = 16000


def str2bool(value: str) -> bool:
    return bool(strtobool(value))


################################################################
# Definition of metrics
################################################################
def dnsmos_metric(model, audio, fs=16000):
    """Calculate the DNSMOS metric.

    Args:
        model (torch.nn.Module): DNSMOS model
        audio (np.ndarray): enhanced signal (time,)
        fs (int): sampling rate in Hz
    Returns:
        dnsmos (float): DNSMOS OVRL value between [1, 5]
    """
    if fs != TARGET_FS:
        audio = soxr.resample(audio, fs, TARGET_FS)
        fs = TARGET_FS
    with torch.no_grad():
        dnsmos_score = model(audio, fs)
    return float(dnsmos_score["OVRL"])


################################################################
# Main entry
################################################################
def main(args):
    data_pairs = []
    with open(args.inf_scp, "r") as f:
        for line in f:
            uid, audio_path = line.strip().split()
            data_pairs.append((uid, audio_path))

    size = len(data_pairs)
    assert 1 <= args.job <= args.nsplits <= size
    interval = size // args.nsplits
    start = (args.job - 1) * interval
    end = size if args.job == args.nsplits else start + interval
    data_pairs = data_pairs[start:end]
    print(
        f"[Job {args.job}/{args.nsplits}] Processing ({len(data_pairs)}/{size}) samples",
        flush=True,
    )
    suffix = "" if args.nsplits == args.job == 1 else f".{args.job}"

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    writers = {
        metric: (outdir / f"{metric}{suffix}.scp").open("w") for metric in METRICS
    }

    if not Path(args.primary_model).exists():
        raise ValueError(
            f"The primary model '{args.primary_model}' doesn't exist."
            " You can download the model from https://github.com/microsoft/"
            "DNS-Challenge/tree/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx"
        )
    if not Path(args.p808_model).exists():
        raise ValueError(
            f"The P808 model '{args.p808_model}' doesn't exist."
            " You can download the model from https://github.com/microsoft/"
            "DNS-Challenge/tree/master/DNSMOS/DNSMOS/model_v8.onnx"
        )

    use_gpu = True if "cuda" in args.device else False
    model = DNSMOS_local(
        args.primary_model,
        args.p808_model,
        use_gpu=use_gpu,
        convert_to_torch=args.convert_to_torch,
    )
    ret = []
    for uid, inf_audio in tqdm(data_pairs):
        _, score = process_one_pair((uid, inf_audio), model=model)
        ret.append((uid, score))
        for metric, value in score.items():
            writers[metric].write(f"{uid} {value}\n")

    for metric in METRICS:
        writers[metric].close()

    if args.nsplits == args.job == 1:
        with (outdir / "RESULTS.txt").open("w") as f:
            for metric in METRICS:
                mean_score = np.nanmean([score[metric] for uid, score in ret])
                f.write(f"{metric}: {mean_score:.4f}\n")
        print(
            f"Overall results have been written in {outdir / 'RESULTS.txt'}", flush=True
        )


def process_one_pair(data_pair, model=None):
    uid, inf_path = data_pair
    inf, fs = sf.read(inf_path, dtype="float32")
    assert inf.ndim == 1, inf.shape

    scores = {}
    for metric in METRICS:
        if metric == "DNSMOS_OVRL":
            scores[metric] = dnsmos_metric(model, inf, fs=fs)
        else:
            raise NotImplementedError(metric)

    return uid, scores


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inf_scp",
        type=str,
        required=True,
        help="Path to the scp file containing enhanced signals",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory for writing metrics",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for running DNSMOS calculation",
    )
    parser.add_argument(
        "--nsplits",
        type=int,
        default=1,
        help="Total number of computing nodes to speed up evaluation",
    )
    parser.add_argument(
        "--job",
        type=int,
        default=1,
        help="Index of the current node (starting from 1)",
    )

    group = parser.add_argument_group("DNSMOS related")
    group.add_argument(
        "--convert_to_torch",
        type=str2bool,
        default=False,
        help="Convert onnx to PyTorch by using onnx2torch",
    )
    group.add_argument(
        "--primary_model",
        type=str,
        default="./DNSMOS/sig_bak_ovr.onnx",
        help="Path to the primary DNSMOS model.",
    )
    group.add_argument(
        "--p808_model",
        type=str,
        default="./DNSMOS/model_v8.onnx",
        help="Path to the p808 model.",
    )
    args = parser.parse_args()

    main(args)
