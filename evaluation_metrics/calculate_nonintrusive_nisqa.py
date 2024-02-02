from distutils.util import strtobool
from pathlib import Path

import numpy as np
from tqdm import tqdm

from nisqa_utils import load_nisqa_model, predict_nisqa


METRICS = ("NISQA_MOS",)


def str2bool(value: str) -> bool:
    return bool(strtobool(value))


################################################################
# Definition of metrics
################################################################
def nisqa_metric(model, audio_path):
    """Calculate the NISQA metric.

    Args:
        model (torch.nn.Module): NISQA model
        audio_path: path to the enhanced signal
    Returns:
        dnsmos (float): NISQA MOS value between [1, 5]
    """
    nisqa_score = predict_nisqa(model, audio_path)
    return float(nisqa_score["mos_pred"])


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

    if not Path(args.nisqa_model).exists():
        raise ValueError(
            f"The NISQA model '{args.nisqa_model}' doesn't exist."
            " You can download the model from https://github.com/gabrielmittag/NISQA"
            "/blob/master/weights/nisqa.tar"
        )

    model = load_nisqa_model(args.nisqa_model, device=args.device)
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

    scores = {}
    for metric in METRICS:
        if metric == "NISQA_MOS":
            scores[metric] = nisqa_metric(model, inf_path)
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

    group = parser.add_argument_group("NISQA related")
    group.add_argument(
        "--nisqa_model",
        type=str,
        default="./lib/NISQA/weights/nisqa.tar",
        help="Path to the pretrained NISQA model (v2.0).",
    )
    args = parser.parse_args()

    main(args)
